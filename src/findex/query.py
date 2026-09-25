import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from findex.tokenize import tokenize

if TYPE_CHECKING:
    from findex.index import Index


class Node(ABC):
    @abstractmethod
    def evaluate(self, index: "Index") -> set[int]: ...

    @abstractmethod
    def terms(self) -> set[str]: ...

    def __and__(self, other: "Node") -> "And":
        return And(self, other)

    def __or__(self, other: "Node") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)


@dataclass(frozen=True)
class Term(Node):
    term: str

    def evaluate(self, index: "Index") -> set[int]:
        term_norm = self.term.casefold()
        if term_norm in index:
            return {p.doc_id for p in index[term_norm]}
        return set()

    def terms(self) -> set[str]:
        return {self.term.casefold()}


@dataclass(frozen=True)
class Phrase(Node):
    phrase_terms: tuple[str, ...]

    def evaluate(self, index: "Index") -> set[int]:
        if not self.phrase_terms:
            return set()

        lists = [index.get_postings(t) for t in self.phrase_terms]
        if any(not l for l in lists):
            return set()

        common_docs = {p.doc_id for p in lists[0]}
        for l in lists[1:]:
            common_docs.intersection_update({p.doc_id for p in l})

        matching_docs: set[int] = set()
        for doc_id in common_docs:
            pos_lists = [
                next(p.positions for p in l if p.doc_id == doc_id) for l in lists
            ]

            for start in pos_lists[0]:
                curr = start
                valid = True
                for next_positions in pos_lists[1:]:
                    curr += 1
                    if curr not in next_positions:
                        valid = False
                        break
                if valid:
                    matching_docs.add(doc_id)
                    break
        return matching_docs

    def terms(self) -> set[str]:
        return {t.casefold() for t in self.phrase_terms}


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index: "Index") -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def terms(self) -> set[str]:
        return self.left.terms() | self.right.terms()


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index: "Index") -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def terms(self) -> set[str]:
        return self.left.terms() | self.right.terms()


@dataclass(frozen=True)
class Not(Node):
    subquery: Node

    def evaluate(self, index: "Index") -> set[int]:
        all_docs = set(range(index.num_docs))
        return all_docs - self.subquery.evaluate(index)

    def terms(self) -> set[str]:
        return set()


TOKEN_RE = re.compile(
    r'\s*(?:(\()|(\))|("(?:\\.|[^"\\])*")|(\bOR\b|\bAND\b|\bNOT\b)|([^\s()]+))',
    re.IGNORECASE,
)


class QueryParser:
    def __init__(self, raw: str) -> None:
        self.tokens: list[tuple[str, str]] = []
        for lparen, rparen, phrase, op, word in TOKEN_RE.findall(raw):
            if lparen:
                self.tokens.append(("LPAREN", "("))
            elif rparen:
                self.tokens.append(("RPAREN", ")"))
            elif phrase:
                # Strip bounding quotes
                clean_phrase = phrase[1:-1]
                self.tokens.append(("PHRASE", clean_phrase))
            elif op:
                self.tokens.append((op.upper(), op.upper()))
            elif word:
                self.tokens.append(("WORD", word))
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos][0] if self.pos < len(self.tokens) else None

    def match(self, expected_type: str) -> str | None:
        if self.peek() == expected_type:
            val = self.tokens[self.pos][1]
            self.pos += 1
            return val
        return None

    def parse(self) -> Node:
        if not self.tokens:
            return Term("")
        node = self.expr_or()
        return node

    def expr_or(self) -> Node:
        left = self.expr_and()
        while self.match("OR"):
            right = self.expr_and()
            left = Or(left, right)
        return left

    def expr_and(self) -> Node:
        left = self.expr_unary()
        while self.peek() not in (None, "OR", "RPAREN"):
            _ = self.match("AND")
            right = self.expr_unary()
            left = And(left, right)
        return left

    def expr_unary(self) -> Node:
        if self.match("NOT"):
            return Not(self.expr_unary())
        return self.expr_primary()

    def expr_primary(self) -> Node:
        if self.match("LPAREN"):
            node = self.expr_or()
            if not self.match("RPAREN"):
                pass
            return node
        phrase_val = self.match("PHRASE")
        if phrase_val is not None:
            terms = tuple(tokenize(phrase_val))
            return Phrase(terms) if len(terms) > 1 else Term(terms[0] if terms else "")
        word_val = self.match("WORD")
        if word_val is not None:
            terms = list(tokenize(word_val))
            return Term(terms[0] if terms else word_val.casefold())
        return Term("")


def parse(query_str: str) -> Node:
    return QueryParser(query_str).parse()
