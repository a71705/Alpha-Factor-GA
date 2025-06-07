import unittest
import random
import re # For testing parse_expression if its regex is complex

# Add project root to sys.path to allow importing project modules
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alpha_models import Node, depth_one_trees, depth_two_tree, depth_three_tree
from alpha_models import d1tree_to_alpha, d2tree_to_alpha, d3tree_to_alpha
from alpha_models import parse_expression, d1_alpha_to_tree, d2_alpha_to_tree, d3_alpha_to_tree
from alpha_models import copy_tree, get_random_node, collect_nodes
from alpha_models import InvalidTreeStructureError, ExpressionParsingError # Custom exceptions
from config import TERMINAL_VALUES, BINARY_OPS, TS_OPS, TS_OPS_VALUES, UNARY_OPS


class TestAlphaModelsNode(unittest.TestCase):
    def test_node_creation(self):
        """测试Node对象是否能正确创建并存储其值。"""
        node = Node("add")
        self.assertEqual(node.value, "add")
        self.assertIsNone(node.left)
        self.assertIsNone(node.right)

        node_with_children = Node("ts_rank")
        left_child = Node("close")
        right_child = Node("20")
        node_with_children.left = left_child
        node_with_children.right = right_child
        self.assertEqual(node_with_children.left.value, "close")
        self.assertEqual(node_with_children.right.value, "20")

class TestAlphaModelsTreeGeneration(unittest.TestCase):
    def test_depth_one_returns_node(self):
        """测试depth_one_trees是否返回一个Node对象。"""
        tree = depth_one_trees()
        self.assertIsInstance(tree, Node)

    def test_depth_one_root_value(self):
        """测试depth_one_trees生成的树的根节点值是否在预期的操作符列表中。"""
        for _ in range(10): # Run a few times due to randomness
            tree = depth_one_trees()
            self.assertTrue(tree.value in BINARY_OPS or tree.value in TS_OPS)
            self.assertIsNotNone(tree.left)
            self.assertIsNotNone(tree.right)
            if tree.value in BINARY_OPS:
                self.assertTrue(tree.left.value in TERMINAL_VALUES)
                self.assertTrue(tree.right.value in TERMINAL_VALUES)
            elif tree.value in TS_OPS:
                self.assertTrue(tree.left.value in TERMINAL_VALUES)
                self.assertTrue(tree.right.value in TS_OPS_VALUES)

    def test_depth_two_returns_node(self):
        """测试depth_two_tree是否返回一个Node对象。"""
        d1_tree1 = depth_one_trees()
        d1_tree2 = depth_one_trees()
        tree = depth_two_tree(d1_tree1, d1_tree2)
        self.assertIsInstance(tree, Node)
        self.assertTrue(tree.value in BINARY_OPS or tree.value in TS_OPS)
        self.assertIsInstance(tree.left, Node) # Left child should be a D1 tree
        if tree.value in BINARY_OPS:
            self.assertIsInstance(tree.right, Node) # Right child is a D1 tree for BINARY_OPS
        elif tree.value in TS_OPS:
            self.assertIsInstance(tree.right, Node) # Right child is a param node for TS_OPS
            self.assertTrue(tree.right.value in TS_OPS_VALUES)


    def test_depth_three_returns_node(self):
        """测试depth_three_tree是否返回一个Node对象。"""
        d1_tree1 = depth_one_trees()
        d1_tree2 = depth_one_trees()
        d2_trees = [depth_two_tree(d1_tree1, d1_tree2) for _ in range(5)]
        tree = depth_three_tree(d2_trees)
        self.assertIsInstance(tree, Node)
        self.assertTrue(tree.value in UNARY_OPS or tree.value in BINARY_OPS or tree.value in TS_OPS)


class TestAlphaModelsCopyTree(unittest.TestCase):
    def test_copy_simple_tree(self):
        """测试简单树的复制。"""
        original = Node("add")
        original.left = Node("close")
        original.right = Node("open")

        copied = copy_tree(original)
        self.assertIsNotNone(copied)
        self.assertEqual(copied.value, "add")
        self.assertIsNotNone(copied.left)
        self.assertEqual(copied.left.value, "close")
        self.assertIsNotNone(copied.right)
        self.assertEqual(copied.right.value, "open")

    def test_copy_is_deep_copy(self):
        """测试复制是深拷贝（不同对象，相同值）。"""
        original = Node("add")
        original.left = Node("close")

        copied = copy_tree(original)
        self.assertNotEqual(id(original), id(copied))
        self.assertNotEqual(id(original.left), id(copied.left))
        self.assertEqual(original.left.value, copied.left.value)

    def test_copy_empty_tree(self):
        """测试复制None是否返回None。"""
        self.assertIsNone(copy_tree(None))

class TestAlphaModelsParseExpression(unittest.TestCase):
    def test_parse_simple_add(self):
        """测试解析简单加法表达式。"""
        expr = ["add(close,open)"]
        expected = [["add", "close", "open"]]
        self.assertEqual(parse_expression(expr), expected)

    def test_parse_ts_ops(self):
        """测试解析时间序列操作符。"""
        expr = ["ts_rank(vwap,20)"]
        expected = [["ts_rank", "vwap", "20"]]
        self.assertEqual(parse_expression(expr), expected)

    def test_parse_numeric_params(self):
        """测试解析带数字参数的表达式。"""
        expr = ["ts_delta(close,10)"]
        expected = [["ts_delta", "close", "10"]]
        self.assertEqual(parse_expression(expr), expected)

    def test_parse_empty_string(self):
        """测试解析空字符串。"""
        self.assertEqual(parse_expression([""]), [[]])

    def test_parse_invalid_expression(self):
        """测试解析无效表达式（例如，只有括号）。"""
        # Current parse_expression filters out brackets, so "()" becomes []
        self.assertEqual(parse_expression(["()"]), [[]])
        self.assertEqual(parse_expression(["(add)"]), [["add"]]) # This is how it currently works

    def test_parse_complex_expression(self):
        """测试解析更复杂的嵌套表达式。"""
        expr = ["subtract(ts_mean(close,20),ts_mean(vwap,40))"]
        # Based on current regex, this is the expected tokenization
        expected = [["subtract", "ts_mean", "close", "20", "ts_mean", "vwap", "40"]]
        self.assertEqual(parse_expression(expr), expected)

    def test_parse_non_string_input(self):
        """测试当输入不是字符串列表时是否抛出TypeError。"""
        with self.assertRaises(ExpressionParsingError): # Or TypeError, depends on implementation
             parse_expression([123])


class TestAlphaModelsTreeToAlpha(unittest.TestCase):
    def test_d1tree_to_alpha_simple(self):
        """测试简单D1树到表达式的转换。"""
        tree = Node("add")
        tree.left = Node("close")
        tree.right = Node("open")
        self.assertEqual(d1tree_to_alpha(tree), "add(close,open)")

        ts_tree = Node("ts_rank")
        ts_tree.left = Node("vwap")
        ts_tree.right = Node("20")
        self.assertEqual(d1tree_to_alpha(ts_tree), "ts_rank(vwap,20)")

    def test_d1tree_to_alpha_invalid(self):
        """测试D1树结构不完整时是否抛出错误。"""
        tree_no_left = Node("add")
        tree_no_left.right = Node("open")
        with self.assertRaises(InvalidTreeStructureError):
            d1tree_to_alpha(tree_no_left)

    def test_d2tree_to_alpha_simple(self):
        """测试简单D2树到表达式的转换。"""
        # Binary op with D1 children
        d1_left = Node("ts_delay"); d1_left.left = Node("close"); d1_left.right = Node("1")
        d1_right = Node("ts_delta"); d1_right.left = Node("open"); d1_right.right = Node("2")
        d2_binary = Node("add"); d2_binary.left = d1_left; d2_binary.right = d1_right
        self.assertEqual(d2tree_to_alpha(d2_binary), "add(ts_delay(close,1),ts_delta(open,2))")

        # TS op with D1 child and param
        d1_child = Node("rank"); d1_child.left = Node("high"); d1_child.right = Node("low") # rank(high,low)
        d2_ts = Node("ts_mean"); d2_ts.left = d1_child; d2_ts.right = Node("20")
        self.assertEqual(d2tree_to_alpha(d2_ts), "ts_mean(rank(high,low),20)")

    def test_d3tree_to_alpha_simple(self):
        """测试简单D3树到表达式的转换。"""
        # Unary op with D2 child
        d1_l = Node("ts_delay"); d1_l.left = Node("close"); d1_l.right = Node("1")
        d1_r = Node("ts_delta"); d1_r.left = Node("open"); d1_r.right = Node("2")
        d2_child = Node("add"); d2_child.left = d1_l; d2_child.right = d1_r # add(ts_delay(close,1),ts_delta(open,2))
        d3_unary = Node("log"); d3_unary.left = d2_child
        self.assertEqual(d3tree_to_alpha(d3_unary), "log(add(ts_delay(close,1),ts_delta(open,2)))")


class TestAlphaModelsAlphaToTree(unittest.TestCase):
    # These tests are more challenging due to the simplified nature of _build_tree_from_components
    # and the fact that d3_alpha_to_tree's inverse logic is not fully implemented.
    def test_d1_alpha_to_tree_simple(self):
        """测试简单D1表达式到树的转换。"""
        alphas = ["add(close,open)"]
        trees = d1_alpha_to_tree(alphas)
        self.assertEqual(len(trees), 1)
        self.assertEqual(trees[0].value, "add")
        self.assertEqual(trees[0].left.value, "close")
        self.assertEqual(trees[0].right.value, "open")

    def test_d2_alpha_to_tree_simple_ts(self):
        """测试简单D2 TS表达式到树的转换。"""
        alphas = ["ts_rank(add(close,open),20)"] # ts_rank(d1_tree, param)
        # Expected components from parse_expression: ['ts_rank', 'add', 'close', 'open', '20']
        trees = d2_alpha_to_tree(alphas)
        self.assertEqual(len(trees), 1)
        tree = trees[0]
        self.assertEqual(tree.value, "ts_rank")
        self.assertIsInstance(tree.left, Node)
        self.assertEqual(tree.left.value, "add")
        self.assertEqual(tree.left.left.value, "close")
        self.assertEqual(tree.left.right.value, "open")
        self.assertIsInstance(tree.right, Node)
        self.assertEqual(tree.right.value, "20")
        self.assertIsNone(tree.right.left) # Param node should be a leaf

    def test_d3_alpha_to_tree_placeholder(self):
        """占位符测试，因为D3解析未完全实现。"""
        # Current d3_alpha_to_tree returns empty list due to unimplemented _build_tree_from_components['d3']
        alphas = ["log(add(ts_delay(close,1),ts_delta(open,2)))"]
        trees = d3_alpha_to_tree(alphas)
        self.assertEqual(len(trees), 0) # Expecting it to fail gracefully for now


class TestAlphaModelsTreeUtils(unittest.TestCase):
    def setUp(self):
        # Simple tree for testing: add(close, open)
        self.tree = Node("add")
        self.tree.left = Node("close")
        self.tree.right = Node("open")
        # Slightly more complex: ts_rank(add(close,open),20)
        self.tree2 = Node("ts_rank")
        self.tree2.left = self.tree # D1 tree as left child
        self.tree2.right = Node("20")


    def test_collect_nodes(self):
        """测试 collect_nodes 是否能收集树中的所有节点。"""
        nodes = []
        collect_nodes(self.tree, nodes)
        self.assertEqual(len(nodes), 3)
        node_values = [n.value for n in nodes]
        self.assertIn("add", node_values)
        self.assertIn("close", node_values)
        self.assertIn("open", node_values)

        nodes2 = []
        collect_nodes(self.tree2, nodes2) # ts_rank(add(close,open),20)
        self.assertEqual(len(nodes2), 5) # ts_rank, add, close, open, 20
        node_values2 = [n.value for n in nodes2]
        self.assertIn("ts_rank", node_values2)
        self.assertIn("add", node_values2)
        self.assertIn("20", node_values2)


    def test_get_random_node(self):
        """测试 get_random_node 是否返回树中的一个有效节点。"""
        if self.tree: # Ensure tree is not None
            random_node = get_random_node(self.tree)
            self.assertIsNotNone(random_node)
            self.assertIn(random_node.value, ["add", "close", "open"])

        self.assertIsNone(get_random_node(None))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

# TODO: Add more tests for edge cases and complex structures, especially for d*tree_to_alpha and d*_alpha_to_tree.
# TODO: Test robustness of tree generation functions when operator/terminal lists are empty (they should raise ValueError).
# TODO: Test robustness of tree-to-alpha and alpha-to-tree functions with unknown operators.
# TODO: Test d3_alpha_to_tree more thoroughly once its parsing logic is fully implemented.
