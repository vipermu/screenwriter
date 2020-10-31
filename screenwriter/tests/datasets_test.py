import unittest

from transformers import GPT2Tokenizer

from screenwritersets import ScreenwriterData

class DatasetTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_remove_pagination(self):
        input_text_list = [
            "71   Bla bla bla and blabla",
            "Bla bla bla and blabla    82",
            "0  Bla bla bla and blabla    822",
            "      Bla bla bla and blabla   ",
            "8. Bla bla bla and blabla   ",
            "Bla bla bla and blabla 1882.",
            "8 Bla bla bla and blabla   ",
            "Bla bla bla and blabla 1882",
            "8 Bla bla bla and blabla 1882",
        ]

        gt_text_list = [
            "Bla bla bla and blabla",
            "Bla bla bla and blabla",
            "Bla bla bla and blabla",
            "      Bla bla bla and blabla   ",
            " Bla bla bla and blabla   ",
            "Bla bla bla and blabla ",
            "8 Bla bla bla and blabla   ",
            "Bla bla bla and blabla 1882",
            "8 Bla bla bla and blabla 1882",
        ]

        for input_text, gt_text in zip(input_text_list, gt_text_list):
            processed_text = ScreenwriterData.remove_pagination(input_text)
            self.assertEqual(processed_text, gt_text)
    
    def test_recognize_dialog(self):
        input_text_list = [
            "                          JIM",
            "JIM",
            " BLABLA:",
        ]
        gt_result_list = [
            True,
            True,
            False,
        ]

        for input_text, gt_result in zip(input_text_list, gt_result_list):
            is_dialog = ScreenwriterData.recognize_dialog(input_text)
            self.assertEqual(is_dialog, gt_result)
        
if __name__ == "__main__":
    suite = unittest.TestSuite()
    # suite.addTest(DatasetTest("test_remove_pagination"))
    suite.addTest(DatasetTest("test_recognize_dialog"))

    runner = unittest.TextTestRunner()
    runner.run(suite)
