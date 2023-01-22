import unittest


class TestSpace(unittest.TestCase):

    def setUp(self):
        # this method is called before every test
        pass

    def tearDown(self) -> None:
        # this method is called after every test
        pass

    @unittest.skip("TODO: implement test")
    def test___init__(self):
        pass

    @unittest.skip("TODO: implement test")
    def test_update_adjacency_matrix_for_all_plls_potentially_receiving_a_signal(self):
        pass


if __name__ == '__main__':
    unittest.main()