import utils
import pandas as pd

def test_read_file_exists():
    """Prueba la función read_file cuando el archivo existe."""
    file_path = "test_data.csv"
    df = utils.read_file(file_path)
    assert df is None

def test_read_file_not_exists():
    """Prueba la función read_file cuando el archivo no existe."""
    file_path = "archivo_que_no_existe.csv"
    df = utils.read_file(file_path)
    assert df is None

def test_write_file_success():
    """Prueba la función write_file cuando se escribe correctamente."""
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    file_path = "test_write.csv"
    assert utils.write_file(data, file_path) is True

def test_write_file_failure():
    """Prueba la función write_file cuando falla la escritura."""
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    file_path = "non_existing_folder/test_write.csv"
    assert utils.write_file(data, file_path) is False

def test_write_file_empty_data():
    """Prueba la función write_file con datos vacíos."""
    data = pd.DataFrame()
    file_path = "test_write_empty.csv"
    assert utils.write_file(data, file_path) is False
