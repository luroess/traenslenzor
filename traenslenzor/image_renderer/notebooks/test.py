# %%

from pydantic import BaseModel
from typing_extensions import Literal

print("hello world")


class Point2D(BaseModel):
    x: float
    y: float
    type: Literal["2D"] = "2D"


class Point3D(BaseModel):
    x: float
    y: float
    z: float
    type: Literal["3D"] = "3D"


p = Point2D(x=1.0, y=2.0)
print(p.model_dump(exclude={"type"}))


p3d = Point3D(**p.model_dump(exclude={"type"}), z=3.0)
print(p3d)
