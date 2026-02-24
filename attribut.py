import tkinter as tk
from tkinter import ttk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from networkx.drawing.nx_pydot import graphviz_layout


class GVWNode:
    def __init__(self, name, node_type='input'):
        self.name = name
        self.type = node_type
        self.value = None
        self.inputs = []

    def evaluate(self):
        if self.type == 'input':
            return self.value
        elif self.type == 'and':
            return all(inp.evaluate() for inp in self.inputs)
        elif self.type == 'or':
            return any(inp.evaluate() for inp in self.inputs)


class GVWCircuit:
    def __init__(self):
        self.nodes = {}
        self.graph = nx.DiGraph()
        self.root = None
        self.evaluated_path = set()

    def add_input(self, name):
        node = GVWNode(name, 'input')
        self.nodes[name] = node
        self.graph.add_node(name, label=name)
        return node

    def add_gate(self, name, gate_type, input_names):
        node = GVWNode(name, gate_type)
        for in_name in input_names:
            input_node = self.nodes[in_name]
            node.inputs.append(input_node)
            self.graph.add_edge(in_name, name)
        self.nodes[name] = node
        self.graph.add_node(name, label=gate_type.upper())
        return node

    def set_root(self, name):
        self.root = self.nodes[name]

    def evaluate(self):
        self.evaluated_path = set()

        def trace(node):
            if node.type == 'input':
                return node.value
            elif node.type == 'and':
                result = True
                for inp in node.inputs:
                    r = trace(inp)
                    if r:
                        self.evaluated_path.add((inp.name, node.name))
                    else:
                        result = False
                return result
            elif node.type == 'or':
                result = False
                for inp in node.inputs:
                    r = trace(inp)
                    if r:
                        self.evaluated_path.add((inp.name, node.name))
                        result = True
                return result

        return trace(self.root)

    def draw(self, canvas_frame):
        fig, ax = plt.subplots(figsize=(12, 6))
        pos = nx.nx_pydot.graphviz_layout(self.graph, prog="dot")  

        edge_colors = []
        for u, v in self.graph.edges():
            if (u, v) in self.evaluated_path:
                edge_colors.append('green')
            else:
                edge_colors.append('gray')

        labels = nx.get_node_attributes(self.graph, 'label')
        nx.draw(self.graph, pos, with_labels=True, labels=labels,
                node_color='lightblue', node_size=2200, font_size=10,
                edge_color=edge_colors, width=2.5, ax=ax)

        ax.set_title("Circuit GVW aligné horizontalement")
        fig.tight_layout()

        for widget in canvas_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)




class GVWApp:
    def __init__(self, root):
        self.root = root
        root.title("GVW - Chiffrement par Attribut (8 Attributs)")
        self.circuit = GVWCircuit()
        self.inputs = {}
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.build_circuit()
        self.build_controls()

    def build_circuit(self):
        # Ajouter 8 entrées
        for i in range(1, 9):
            self.circuit.add_input(f'A{i}')

        # ((A1 AND A2) OR (A3 AND A4)) AND ((A5 AND A6) OR (A7 AND A8))
        self.circuit.add_gate('AND1', 'and', ['A1', 'A2'])
        self.circuit.add_gate('AND2', 'and', ['A3', 'A4'])
        self.circuit.add_gate('OR1', 'or', ['AND1', 'AND2'])

        self.circuit.add_gate('AND3', 'and', ['A5', 'A6'])
        self.circuit.add_gate('AND4', 'and', ['A7', 'A8'])
        self.circuit.add_gate('OR2', 'or', ['AND3', 'AND4'])

        self.circuit.add_gate('ROOT', 'and', ['OR1', 'OR2'])

        self.circuit.set_root('ROOT')

    def build_controls(self):
        ttk.Label(self.control_frame, text="Activer les attributs :").pack(pady=10)

        for i in range(1, 9):
            name = f'A{i}'
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(self.control_frame, text=name, variable=var)
            chk.pack(anchor='w')
            self.inputs[name] = var

        ttk.Button(self.control_frame, text="Évaluer le circuit", command=self.evaluate).pack(pady=20)

        self.result_label = ttk.Label(self.control_frame, text="Résultat: ...")
        self.result_label.pack(pady=10)

    def evaluate(self):
        for name, var in self.inputs.items():
            self.circuit.nodes[name].value = var.get()

        result = self.circuit.evaluate()
        self.result_label.config(text=f"Résultat: {' Déchiffrement autorisé' if result else ' Refusé'}")
        self.circuit.draw(self.canvas_frame)


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("1200x700")
    app = GVWApp(root)
    root.mainloop()
