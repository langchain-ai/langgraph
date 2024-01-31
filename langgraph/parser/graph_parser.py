import argparse
import ast
import json
import os
import subprocess


# Path to the config file with the mapping between the workflow nodes and the icons
CONFIG_FILE = os.path.join(
    os.path.dirname(__file__), "config", "config.json"
)


class WorkflowVisitor(ast.NodeVisitor):
    def __init__(self):
        self.edges = []
        self.conditions = {}
        self.entry_point = None
        self.in_workflow = False

    def visit_Assign(self, node):
        if (
            isinstance(node.value, ast.Call)
            and hasattr(node.value.func, "id")
            and node.value.func.id == "StateGraph"
        ):
            self.in_workflow = True
        self.generic_visit(node)

    def visit_Call(self, node):
        if self.in_workflow:
            try:
                if hasattr(node.func, "attr"):
                    func_attr = node.func.attr
                    if func_attr == "compile":
                        self.in_workflow = False
                    elif func_attr == "add_edge":
                        self._handle_add_edge(node)
                    elif func_attr == "add_conditional_edges":
                        self._handle_add_conditional_edges(node)
                    elif func_attr == "set_entry_point":
                        self._handle_set_entry_point(node)
            except Exception as visit_error:
                raise visit_error(f"Error while visiting call node: {visit_error}")
        self.generic_visit(node)

    def _handle_add_edge(self, node):
        if len(node.args) == 2 and all(isinstance(arg, ast.Str) for arg in node.args):
            source, target = [arg.s for arg in node.args]
            self.edges.append((source, target, None))
        else:
            raise ValueError("Invalid arguments for 'add_edge'")

    def _handle_add_conditional_edges(self, node):
        if (
            len(node.args) >= 3
            and isinstance(node.args[0], ast.Str)
            and isinstance(node.args[2], ast.Dict)
        ):
            source = node.args[0].s
            diamond_node = f"{source}_reason"
            self.edges.append((source, diamond_node, None))
            for key, value in zip(node.args[2].keys, node.args[2].values):
                if isinstance(key, ast.Str) and (
                    isinstance(value, ast.Str) or hasattr(value, "id")
                ):
                    condition = key.s
                    target = value.s if isinstance(value, ast.Str) else value.id
                    self.conditions[(diamond_node, target)] = condition
        else:
            raise ValueError("Invalid arguments for 'add_conditional_edges'")

    def _handle_set_entry_point(self, node):
        if len(node.args) == 1 and isinstance(node.args[0], ast.Str):
            self.entry_point = node.args[0].s
        else:
            raise ValueError("Invalid arguments for 'set_entry_point'")

    def load_config(config_file):
        with open(config_file, "r", encoding="utf-8") as file:
            return json.load(file)

    def _is_reason_node(self, node_name):
        return "_reason" in node_name

    def _get_node_representation(self, node_name, config):
        if node_name in ["start", "END"]:
            icon = config["mermaid"]["icons"].get(node_name.lower(), "")
            if node_name == "start":
                label = "start -"
            if node_name == "END":
                label = "end -"
            return f"{node_name}[{icon} {label}]"
        elif "_reason" in node_name:
            # For "reason" nodes, use a different shape
            icon = config["mermaid"]["icons"].get(node_name, node_name)
            label = "reason -"
            return f"{node_name}{{{icon} {label}}}"
        elif "agent" in node_name:
            icon = config["mermaid"]["icons"]["agent"]
            label = f"{node_name} -"
            return f"{node_name}[{icon} {label}]"
        elif "action" in node_name:
            icon = config["mermaid"]["icons"]["action"]
            label = f"{node_name} -"
            return f"{node_name}[{icon} {label}]"


def generate_mermaid_code(graph_file_path):
    try:
        with open(graph_file_path, "r", encoding="utf-8") as file:
            tree = ast.parse(file.read())
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {graph_file_path}")
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in the file: {e}")
    except Exception as mermaid_code_error:
        raise mermaid_code_error(f"Error parsing the file: {mermaid_code_error}")

    visitor = WorkflowVisitor()
    try:
        visitor.visit(tree)
    except Exception as visit_error:
        log.error(f"An error occurred while visiting the AST: {visit_error}")
        raise  visit_error(f"An error occurred while visiting the AST: {visit_error}")
    
    config = WorkflowVisitor.load_config(CONFIG_FILE)

    mermaid_output_code = "graph TD\n"
    if visitor.entry_point:
        entry_point_repr = visitor._get_node_representation(visitor.entry_point, config)
        start_repr = visitor._get_node_representation("start", config)
        mermaid_output_code += f"    {start_repr} --> {entry_point_repr}\n"
    for source, target, _ in visitor.edges:
        source_repr = visitor._get_node_representation(source, config)
        target_repr = visitor._get_node_representation(target, config)
        mermaid_output_code += f"    {source_repr} --> {target_repr}\n"
    for (source, target), condition in visitor.conditions.items():
        source_repr = visitor._get_node_representation(source, config)
        target_repr = visitor._get_node_representation(target, config)
        mermaid_output_code += f"    {source_repr} -->|{condition}| {target_repr}\n"

    return mermaid_output_code


def generate_mermaid_png(mermaid_code, png_path):
    temp_mermaid_file = "temp.mmd"
    theme = "default"  # [default, forest, dark, neutral]
    css_file = "mermaid.css"
    config_file = "mermaid-config.json"
    with open(temp_mermaid_file, "w", encoding="utf-8") as file:
        file.write(mermaid_code)

    # Prepare Docker command
    docker_command = [
        "docker-compose",
        "run",
        "--rm",
        "mermaid-cli",
        "-i",
        f"/data/{temp_mermaid_file}",
        "-t",
        f"{theme}",
        "--cssFile",
        f"/data/{css_file}",
        "--configFile",
        f"/data/{config_file}",
        "-o",
        f"/data/{png_path}",
    ]

    try:
        subprocess.run(docker_command, check=True)
    except subprocess.CalledProcessError:
        raise subprocess.CalledProcessError(
            f"Error in Docker process: {subprocess.CalledProcessError}"
        )
    except Exception as subprocess_error:
        raise subprocess_error(
            f"Unexpected error: {subprocess_error}"
        )
    finally:
        if os.path.exists(temp_mermaid_file):
            os.remove(temp_mermaid_file)
        subprocess.run(["docker-compose", "down"], check=True)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate Mermaid diagram from LangGraph framework."
    )
    parser.add_argument("file_path", help="Path to the Python script to be parsed.")
    parser.add_argument("-o", "--output", help="Output PNG file name.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    file_path = args.file_path
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_png_path = args.output if args.output else f"{base_name}.png"

    try:
        generated_mermaid_code = generate_mermaid_code(file_path)
        generate_mermaid_png(generated_mermaid_code, output_png_path)
    except Exception as main_error:
        raise
