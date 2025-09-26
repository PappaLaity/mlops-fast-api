from pydantic import BaseModel, Field


class IrisData(BaseModel):
    sepalLength: float = Field(title="Sepal length in cm", gt=0, le=8.0, default=5.1)
    sepalWidth: float = Field(title="Sepal Width in cm", gt=0, le=4.5, default=3.5)
    petalLength: float = Field(title="Petal length in cm", gt=0, le=7.0, default=1.4)
    petalWidth: float = Field(title="Petal Width in cm", gt=0, le=2.5, default=0.2)
    # species : str | None = None
    # models : str | None = None