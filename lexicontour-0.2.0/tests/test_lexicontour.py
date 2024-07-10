from lexicontour import lexicontourfunctions
from pathlib import Path

def test_contours():
   assert len(lexicontourfunctions.get_dict('C:\\Users\\admin\\Desktop\\PROG\\instance\\uploads\\INVOICE@LCOA.PDF')) !=0, "Error!"