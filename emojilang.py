import sys
from dataclasses import dataclass
from typing import List, Union

# Symbol table for variables
variables = {}

# Token types
class TokenType:
    DECL = "DECL"  # 🆕
    PRINT = "PRINT"  # 💬
    INPUT = "INPUT"  # ⌨️
    IF = "IF"  # ❓
    ELSE = "ELSE"  # ❗
    LOOP = "LOOP"  # 🔄
    BLOCK_START = "BLOCK_START"  # ▶️
    BLOCK_END = "BLOCK_END"  # ⏹️
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    STRING = "STRING"
    EQUALS = "EQUALS"  # =
    PLUS = "PLUS"  # +
    MINUS = "MINUS"  # -
    MULT = "MULT"  # *
    DIV = "DIV"  # /
    GT = "GT"  # >
    LT = "LT"  # <
    EQ = "EQ"  # ==
    EOF = "EOF"

@dataclass
class Token:
    type: str
    value: str
    line: int

# AST Nodes
@dataclass
class ASTNode:
    pass

@dataclass
class Program(ASTNode):
    statements: List[ASTNode]

@dataclass
class DeclStmt(ASTNode):
    var: str
    expr: ASTNode

@dataclass
class AssignStmt(ASTNode):
    var: str
    expr: ASTNode

@dataclass
class PrintStmt(ASTNode):
    expr: ASTNode

@dataclass
class InputStmt(ASTNode):
    var: str

@dataclass
class IfStmt(ASTNode):
    condition: ASTNode
    then_block: List[ASTNode]
    else_block: List[ASTNode]

@dataclass
class LoopStmt(ASTNode):
    count: ASTNode
    block: List[ASTNode]

@dataclass
class NumberExpr(ASTNode):
    value: int

@dataclass
class StringExpr(ASTNode):
    value: str

@dataclass
class VarExpr(ASTNode):
    name: str

@dataclass
class BinaryExpr(ASTNode):
    left: ASTNode
    op: str
    right: ASTNode

# Lexer
class Lexer:
    def __init__(self, code: str):
        self.lines = code.split('\n')
        self.line_num = 0
        self.current_tokens = []
        self.token_pos = 0

    def next_token(self) -> Token:
        while True:
            if self.token_pos < len(self.current_tokens):
                token = self.current_tokens[self.token_pos]
                self.token_pos += 1
                return token
            else:
                if self.line_num >= len(self.lines):
                    return Token(TokenType.EOF, "", self.line_num)
                line = self.lines[self.line_num].strip()
                self.line_num += 1
                if line:
                    self.current_tokens = self.tokenize_line(line)
                    self.token_pos = 0

    def tokenize_line(self, line: str) -> List[Token]:
        tokens = []
        current = ''
        in_quote = False
        i = 0

        while i < len(line):
            char = line[i]
            if char == '"':
                if in_quote:
                    current += char
                    tokens.append(Token(TokenType.STRING, current, self.line_num - 1))
                    current = ''
                    in_quote = False
                else:
                    if current:
                        tokens.append(self.classify_token(current))
                    current = char
                    in_quote = True
                i += 1
            elif in_quote:
                current += char
                i += 1
            elif char.isspace():
                if current:
                    tokens.append(self.classify_token(current))
                    current = ''
                i += 1
            else:
                current += char
                i += 1

        if current:
            tokens.append(self.classify_token(current))
        return tokens

    def classify_token(self, value: str) -> Token:
        emoji_map = {
            '🆕': TokenType.DECL,
            '💬': TokenType.PRINT,
            '⌨️': TokenType.INPUT,
            '❓': TokenType.IF,
            '❗': TokenType.ELSE,
            '🔄': TokenType.LOOP,
            '▶️': TokenType.BLOCK_START,
            '⏹️': TokenType.BLOCK_END,
        }
        op_map = {
            '=': TokenType.EQUALS,
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.MULT,
            '/': TokenType.DIV,
            '>': TokenType.GT,
            '<': TokenType.LT,
            '==': TokenType.EQ,
        }

        if value in emoji_map:
            return Token(emoji_map[value], value, self.line_num - 1)
        elif value in op_map:
            return Token(op_map[value], value, self.line_num - 1)
        elif value.startswith('"') and value.endswith('"'):
            return Token(TokenType.STRING, value, self.line_num - 1)
        elif value.isdigit():
            return Token(TokenType.NUMBER, value, self.line_num - 1)
        elif value.isidentifier():
            return Token(TokenType.IDENTIFIER, value, self.line_num - 1)
        else:
            raise SyntaxError(f"Invalid token '{value}' at line {self.line_num - 1}")

# Parser
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current_token(self) -> Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else Token(TokenType.EOF, "", -1)

    def consume(self, expected_type: str):
        token = self.current_token()
        if token.type != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {token.type} at line {token.line}")
        self.pos += 1
        return token

    def parse(self) -> Program:
        statements = []
        while self.current_token().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        return Program(statements)

    def parse_statement(self) -> Union[ASTNode, None]:
        token = self.current_token()
        if token.type == TokenType.DECL:
            return self.parse_decl_stmt()
        elif token.type == TokenType.IDENTIFIER:
            return self.parse_assign_stmt()
        elif token.type == TokenType.PRINT:
            return self.parse_print_stmt()
        elif token.type == TokenType.INPUT:
            return self.parse_input_stmt()
        elif token.type == TokenType.IF:
            return self.parse_if_stmt()
        elif token.type == TokenType.LOOP:
            return self.parse_loop_stmt()
        elif token.type in (TokenType.BLOCK_START, TokenType.BLOCK_END, TokenType.ELSE):
            raise SyntaxError(f"Unexpected {token.type} at line {token.line}")
        else:
            self.pos += 1  # Skip empty or invalid
            return None

    def parse_decl_stmt(self) -> DeclStmt:
        self.consume(TokenType.DECL)
        var = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.EQUALS)
        expr = self.parse_expression()
        return DeclStmt(var, expr)

    def parse_assign_stmt(self) -> AssignStmt:
        var = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.EQUALS)
        expr = self.parse_expression()
        return AssignStmt(var, expr)

    def parse_print_stmt(self) -> PrintStmt:
        self.consume(TokenType.PRINT)
        expr = self.parse_expression()
        return PrintStmt(expr)

    def parse_input_stmt(self) -> InputStmt:
        self.consume(TokenType.INPUT)
        var = self.consume(TokenType.IDENTIFIER).value
        return InputStmt(var)

    def parse_if_stmt(self) -> IfStmt:
        self.consume(TokenType.IF)
        condition = self.parse_expression()
        self.consume(TokenType.BLOCK_START)
        then_block = []
        while self.current_token().type != TokenType.BLOCK_END:
            stmt = self.parse_statement()
            if stmt:
                then_block.append(stmt)
        self.consume(TokenType.BLOCK_END)
        else_block = []
        if self.current_token().type == TokenType.ELSE:
            self.consume(TokenType.ELSE)
            self.consume(TokenType.BLOCK_START)
            while self.current_token().type != TokenType.BLOCK_END:
                stmt = self.parse_statement()
                if stmt:
                    else_block.append(stmt)
            self.consume(TokenType.BLOCK_END)
        return IfStmt(condition, then_block, else_block)

    def parse_loop_stmt(self) -> LoopStmt:
        self.consume(TokenType.LOOP)
        count = self.parse_expression()
        self.consume(TokenType.BLOCK_START)
        block = []
        while self.current_token().type != TokenType.BLOCK_END:
            stmt = self.parse_statement()
            if stmt:
                block.append(stmt)
        self.consume(TokenType.BLOCK_END)
        return LoopStmt(count, block)

    def parse_expression(self) -> ASTNode:
        expr = self.parse_term()
        while self.current_token().type in (TokenType.PLUS, TokenType.MINUS, TokenType.GT, TokenType.LT, TokenType.EQ):
            op = self.current_token().value
            self.pos += 1
            right = self.parse_term()
            expr = BinaryExpr(expr, op, right)
        return expr

    def parse_term(self) -> ASTNode:
        expr = self.parse_factor()
        while self.current_token().type in (TokenType.MULT, TokenType.DIV):
            op = self.current_token().value
            self.pos += 1
            right = self.parse_factor()
            expr = BinaryExpr(expr, op, right)
        return expr

    def parse_factor(self) -> ASTNode:
        token = self.current_token()
        if token.type == TokenType.NUMBER:
            self.pos += 1
            return NumberExpr(int(token.value))
        elif token.type == TokenType.STRING:
            self.pos += 1
            return StringExpr(token.value[1:-1])  # Remove quotes
        elif token.type == TokenType.IDENTIFIER:
            self.pos += 1
            return VarExpr(token.value)
        else:
            raise SyntaxError(f"Invalid expression at line {token.line}")

# Semantic Analyzer
class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = set()

    def analyze(self, node: ASTNode):
        if isinstance(node, Program):
            for stmt in node.statements:
                self.analyze(stmt)
        elif isinstance(node, DeclStmt):
            if node.var in self.symbol_table:
                raise ValueError(f"Variable '{node.var}' already declared")
            self.symbol_table.add(node.var)
            self.analyze(node.expr)
        elif isinstance(node, AssignStmt):
            if node.var not in self.symbol_table:
                raise ValueError(f"Variable '{node.var}' not declared")
            self.analyze(node.expr)
        elif isinstance(node, PrintStmt):
            self.analyze(node.expr)
        elif isinstance(node, InputStmt):
            self.symbol_table.add(node.var) 
        elif isinstance(node, IfStmt):
            self.analyze(node.condition)
            for stmt in node.then_block:
                self.analyze(stmt)
            for stmt in node.else_block:
                self.analyze(stmt)
        elif isinstance(node, LoopStmt):
            self.analyze(node.count)
            for stmt in node.block:
                self.analyze(stmt)
        elif isinstance(node, BinaryExpr):
            self.analyze(node.left)
            self.analyze(node.right)
        elif isinstance(node, VarExpr):
            if node.name not in self.symbol_table:
                raise ValueError(f"Variable '{node.name}' not declared")
        elif isinstance(node, (NumberExpr, StringExpr)):
            pass
        else:
            raise ValueError("Unknown AST node")

# Interpreter
class Interpreter:
    def __init__(self):
        self.variables = {}

    def evaluate(self, node: ASTNode) -> Union[int, str, bool]:
        if isinstance(node, NumberExpr):
            return node.value
        elif isinstance(node, StringExpr):
            return node.value
        elif isinstance(node, VarExpr):
            if node.name not in self.variables:
                raise ValueError(f"Variable '{node.name}' not assigned")
            return self.variables[node.name]
        elif isinstance(node, BinaryExpr):
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            if node.op == '+':
                return left + right
            elif node.op == '-':
                return left - right
            elif node.op == '*':
                return left * right
            elif node.op == '/':
                return left / right
            elif node.op == '>':
                return left > right
            elif node.op == '<':
                return left < right
            elif node.op == '==':
                return left == right
        raise ValueError("Invalid expression")

    def interpret(self, node: ASTNode):
        if isinstance(node, Program):
            for stmt in node.statements:
                self.interpret(stmt)
        elif isinstance(node, DeclStmt):
            value = self.evaluate(node.expr)
            self.variables[node.var] = value
        elif isinstance(node, AssignStmt):
            value = self.evaluate(node.expr)
            self.variables[node.var] = value
        elif isinstance(node, PrintStmt):
            value = self.evaluate(node.expr)
            print(value)
        elif isinstance(node, InputStmt):
            user_input = input("Enter a value: ")
            try:
                value = int(user_input)
            except ValueError:
                value = user_input
            self.variables[node.var] = value
        elif isinstance(node, IfStmt):
            condition = self.evaluate(node.condition)
            if condition:
                for stmt in node.then_block:
                    self.interpret(stmt)
            else:
                for stmt in node.else_block:
                    self.interpret(stmt)
        elif isinstance(node, LoopStmt):
            count = self.evaluate(node.count)
            if not isinstance(count, int):
                raise ValueError("Loop count must be an integer")
            for _ in range(count):
                for stmt in node.block:
                    self.interpret(stmt)

def main():
    if len(sys.argv) < 2:
        print("Usage: python emojilang.py <filename>")
        return

    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        code = f.read()

    # Lexical Analysis
    lexer = Lexer(code)
    tokens = []
    while True:
        token = lexer.next_token()
        tokens.append(token)
        if token.type == TokenType.EOF:
            break

    # Syntax Analysis
    parser = Parser(tokens)
    ast = parser.parse()

    # Semantic Analysis
    semantic_analyzer = SemanticAnalyzer()
    semantic_analyzer.analyze(ast)

    # Execution
    interpreter = Interpreter()
    interpreter.interpret(ast)

if __name__ == "__main__":
    main()