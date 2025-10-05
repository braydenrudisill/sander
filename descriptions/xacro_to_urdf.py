"""Make sure you run this from the annin_ar4 directory with `uv run ../xacro_to_urdf.py`"""

import xacrodoc

xacro_file = "ar.urdf.xacro"
urdf_file = "ar.urdf"

doc = xacrodoc.XacroDoc.from_file(xacro_file, subargs={"ar_model": "mk3"})

urdf_content = doc.to_urdf_string()

with open(urdf_file, "w") as f:
    f.write(urdf_content)

print(f"Successfully converted {xacro_file} to {urdf_file}")