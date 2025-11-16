import sys
import types

# Ugly fix until paddle ocr is langchain 1.0.0 compatible
# https://github.com/PaddlePaddle/PaddleOCR/issues/16711#issuecomment-3446427004

# Provide old import paths expected by paddlex:
# langchain.docstore.document -> Document
m1 = types.ModuleType("langchain.docstore.document")
from langchain_core.documents import Document  # noqa: E402, I001

m1.Document = Document
sys.modules["langchain.docstore.document"] = m1

# langchain.text_splitter -> RecursiveCharacterTextSplitter
m2 = types.ModuleType("langchain.text_splitter")
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402, I001

m2.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter  # type: ignore
sys.modules["langchain.text_splitter"] = m2
