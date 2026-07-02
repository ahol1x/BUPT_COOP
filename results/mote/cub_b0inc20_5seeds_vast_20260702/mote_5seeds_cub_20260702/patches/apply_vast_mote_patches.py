from pathlib import Path

# 1. Fix timm import
p = Path("backbone/linears.py")
s = p.read_text()
s = s.replace(
    "from timm.models.layers.weight_init import trunc_normal_",
    "from timm.layers import trunc_normal_"
)
p.write_text(s)
print("patched backbone/linears.py")

# 2. Make missing CodaPrompt optional
p = Path("utils/inc_net.py")
s = p.read_text()
s = s.replace(
    "from backbone.prompt import CodaPrompt",
    "try:\n    from backbone.prompt import CodaPrompt\nexcept ModuleNotFoundError:\n    CodaPrompt = None"
)
s = s.replace("vit_moe_limit.", "vit_mote_limit.")
p.write_text(s)
print("patched utils/inc_net.py")

# 3. Patch MoTE missing attributes and eval output unpacking
p = Path("models/mote.py")
s = p.read_text()

if "self.moni_adam =" not in s:
    lines = s.splitlines(True)
    new_lines = []
    inserted = False
    attrs = [
        ('moni_adam', 'args.get("moni_adam", False)'),
        ('use_init_ptm', 'args.get("use_init_ptm", False)'),
        ('use_reweight', 'args.get("use_reweight", False)'),
        ('use_old_data', 'args.get("use_old_data", False)'),
        ('alpha', 'args.get("alpha", 0.1)'),
        ('beta', 'args.get("beta", 0)'),
        ('recalc_sim', 'args.get("recalc_sim", True)'),
        ('adapter_num', 'args.get("adapter_num", -1)'),
    ]
    for line in lines:
        new_lines.append(line)
        if "self.args" in line and "=" in line and not inserted:
            indent = line[:len(line) - len(line.lstrip())]
            for name, expr in attrs:
                if f"self.{name} =" not in s:
                    new_lines.append(f"{indent}self.{name} = {expr}\n")
            inserted = True
    if not inserted:
        raise RuntimeError("Could not find self.args assignment in models/mote.py")
    s = "".join(new_lines)

s = s.replace(
    "outputs,_ = self._network.forward(inputs, test=True)",
    "outputs = self._network.forward(inputs, test=True)"
)

p.write_text(s)
print("patched models/mote.py")
