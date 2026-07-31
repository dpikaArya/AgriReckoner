import logging
from typing import Any

import pandas as pd
from pydantic import BaseModel, field_validator


class ColumnSchema(BaseModel):
    name: str
    dtype: str
    nullable: bool = True
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] | None = None
    unit: str | None = None


class TableSchema(BaseModel):
    table_name: str
    columns: list[ColumnSchema]
    required_columns: list[str] | None = None
    primary_key: list[str] | None = None

    @field_validator("required_columns", mode="before")
    @classmethod
    def set_default_required(cls, v, info):
        if v is None and "columns" in info.data:
            return [c.name for c in info.data["columns"] if not c.nullable]
        return v


class SchemaValidator:
    def __init__(self, schema: TableSchema, logger: logging.Logger | None = None):
        self.schema = schema
        self.logger = logger or logging.getLogger(__name__)

    def validate(self, df: pd.DataFrame) -> dict:
        errors = []
        warnings = []
        missing_columns = []
        type_mismatches = []

        schema_col_names = {c.name for c in self.schema.columns}
        df_col_names = set(df.columns)

        for col in self.schema.columns:
            if col.name not in df_col_names:
                missing_columns.append(col.name)
                if col.name in (self.schema.required_columns or []):
                    errors.append(f"Required column '{col.name}' is missing from DataFrame")

        for col_name in df_col_names:
            if col_name not in schema_col_names:
                warnings.append(f"Unexpected column '{col_name}' found in DataFrame")

        expected_dtype_map = {
            "int64": "int64",
            "float64": "float64",
            "object": "object",
            "datetime64": "datetime64[ns]",
        }

        for col in self.schema.columns:
            if col.name not in df_col_names:
                continue
            actual_dtype = str(df[col.name].dtype)
            expected_dtype = expected_dtype_map.get(col.dtype)
            if expected_dtype and actual_dtype != expected_dtype:
                type_mismatches.append(
                    {
                        "column": col.name,
                        "expected": expected_dtype,
                        "actual": actual_dtype,
                    }
                )
                if not col.nullable:
                    errors.append(
                        f"Column '{col.name}' expects dtype {expected_dtype} but got {actual_dtype}"
                    )

        if not col.nullable:
            for col in self.schema.columns:
                if col.name in df_col_names and df[col.name].isnull().any():
                    errors.append(f"Non-nullable column '{col.name}' contains null values")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "missing_columns": missing_columns,
            "type_mismatches": type_mismatches,
        }
