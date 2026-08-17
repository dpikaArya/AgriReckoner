import pandas as pd

from src.validation.schema_validator import ColumnSchema, SchemaValidator, TableSchema


class TestColumnSchema:
    def test_column_schema_creation(self):
        col = ColumnSchema(name="temperature", dtype="float64")
        assert col.name == "temperature"
        assert col.dtype == "float64"
        assert col.nullable is True
        assert col.min_value is None
        assert col.max_value is None
        assert col.allowed_values is None
        assert col.unit is None


class TestTableSchema:
    def test_table_schema_creation(self):
        cols = [
            ColumnSchema(name="temp", dtype="float64", nullable=False),
            ColumnSchema(name="rainfall", dtype="float64", nullable=True),
            ColumnSchema(name="city", dtype="object", nullable=True),
        ]
        schema = TableSchema(
            table_name="weather", columns=cols, primary_key=["temp"], required_columns=["temp"]
        )
        assert schema.table_name == "weather"
        assert len(schema.columns) == 3
        assert schema.primary_key == ["temp"]
        assert schema.required_columns == ["temp"]


class TestSchemaValidator:
    def test_validate_valid_dataframe(self):
        cols = [
            ColumnSchema(name="temp", dtype="float64"),
            ColumnSchema(name="city", dtype="object"),
        ]
        schema = TableSchema(table_name="test", columns=cols)
        validator = SchemaValidator(schema=schema)

        df = pd.DataFrame({"temp": [1.0, 2.0], "city": ["a", "b"]})
        result = validator.validate(df)
        assert result["valid"] is True
        assert result["errors"] == []
        assert result["warnings"] == []

    def test_validate_missing_column(self):
        cols = [
            ColumnSchema(name="temp", dtype="float64"),
            ColumnSchema(name="rainfall", dtype="float64", nullable=False),
        ]
        schema = TableSchema(table_name="test", columns=cols, required_columns=["rainfall"])
        validator = SchemaValidator(schema=schema)

        df = pd.DataFrame({"temp": [1.0, 2.0]})
        result = validator.validate(df)
        assert result["valid"] is False
        assert "rainfall" in result["missing_columns"]
        assert any("rainfall" in e for e in result["errors"])

    def test_validate_type_mismatch(self):
        cols = [
            ColumnSchema(name="temp", dtype="int64", nullable=False),
        ]
        schema = TableSchema(table_name="test", columns=cols)
        validator = SchemaValidator(schema=schema)

        df = pd.DataFrame({"temp": [1.0, 2.0]})
        result = validator.validate(df)
        assert result["valid"] is False
        assert len(result["type_mismatches"]) > 0
        mismatch = result["type_mismatches"][0]
        assert mismatch["column"] == "temp"
        assert mismatch["expected"] == "int64"
        assert "float64" in mismatch["actual"]

    def test_unexpected_column_warning(self):
        cols = [ColumnSchema(name="temp", dtype="float64")]
        schema = TableSchema(table_name="test", columns=cols)
        validator = SchemaValidator(schema=schema)

        df = pd.DataFrame({"temp": [1.0], "extra": ["x"]})
        result = validator.validate(df)
        assert result["valid"] is True
        assert any("extra" in w for w in result["warnings"])
