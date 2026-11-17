"""安全的数学表达式计算工具。"""

from __future__ import annotations

import ast
import operator
from typing import Callable


# 允许使用的二元运算符
_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,          # +
    ast.Sub: operator.sub,          # -
    ast.Mult: operator.mul,         # *
    ast.Div: operator.truediv,      # /
    ast.FloorDiv: operator.floordiv,  # //
    ast.Mod: operator.mod,          # %
    ast.Pow: operator.pow,          # **
}

# 允许使用的一元运算符
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,         # +3
    ast.USub: operator.neg,         # -3
}

_MAX_EXPRESSION_LENGTH = 200
_MAX_POWER = 20
_MAX_ABSOLUTE_RESULT = 1e100


def _evaluate_node(node: ast.AST) -> float:
    """递归计算抽象语法树中的节点。"""

    # 整个表达式的根节点
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)

    # 普通数字，如 123、3.14
    if isinstance(node, ast.Constant):
        value = node.value

        # bool是int的子类，所以需要单独排除
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("表达式中只能出现数字。")

        return float(value)

    # 二元运算，例如 1 + 2
    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)

        if operator_type not in _BINARY_OPERATORS:
            raise ValueError(
                f"不支持运算符：{operator_type.__name__}"
            )

        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)

        if operator_type is ast.Pow and abs(right) > _MAX_POWER:
            raise ValueError(
                f"指数绝对值不能超过 {_MAX_POWER}。"
            )

        try:
            result = _BINARY_OPERATORS[operator_type](
                left,
                right,
            )
        except ZeroDivisionError as exc:
            raise ValueError("除数不能为0。") from exc

        if abs(result) > _MAX_ABSOLUTE_RESULT:
            raise ValueError("计算结果过大。")

        return float(result)

    # 一元运算，例如 -5
    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)

        if operator_type not in _UNARY_OPERATORS:
            raise ValueError(
                f"不支持一元运算符：{operator_type.__name__}"
            )

        value = _evaluate_node(node.operand)
        return float(_UNARY_OPERATORS[operator_type](value))

    raise ValueError(
        "表达式中包含不允许的内容，"
        "目前只支持数字、括号和基本四则运算。"
    )


def calculator(expression: str) -> str:
    """
    计算数学表达式。

    支持：
    加、减、乘、除、整除、取余、乘方和括号。

    Args:
        expression: 待计算的数学表达式。

    Returns:
        字符串形式的计算结果。
    """
    expression = expression.strip()

    if not expression:
        raise ValueError("计算表达式不能为空。")

    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"表达式长度不能超过 {_MAX_EXPRESSION_LENGTH} 个字符。"
        )

    try:
        syntax_tree = ast.parse(
            expression,
            mode="eval",
        )
    except SyntaxError as exc:
        raise ValueError(
            f"表达式格式不正确：{expression}"
        ) from exc

    result = _evaluate_node(syntax_tree)

    # 整数结果不显示 .0
    if result.is_integer():
        return str(int(result))

    return str(result)


# 提供给大模型看的工具说明
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "精确计算数学表达式。"
            "当用户要求执行加减乘除、括号、乘方、"
            "取余或其他基础算术时，应调用该工具，"
            "不要自行心算。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "使用Python数学运算符表示的表达式，"
                        "例如：(123 + 456) * 7"
                    ),
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}