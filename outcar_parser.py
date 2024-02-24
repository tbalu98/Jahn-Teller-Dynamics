import re


file = open("m1ex.OUTCAR", "r")
txt = file.read()
#print(txt)
file.close()
#txt= "0.2 4.3"
#res = re.findall(r"POSITION\s*TOTAL-FORCE \(eV/Angst\)\n\s-*\n(\s|\d|([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?)*-*",txt)
#print(res)
def use_regex(input_text):
    pattern = re.compile(r".*POSITION.*\n \-*[\n|\s|\+|\-|.|\[0-9\]]*", re.IGNORECASE)
    return pattern.findall(input_text)


def find_floating_point(input_text):
    pattern = re.compile(r"([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?", re.IGNORECASE)
    return pattern.findall(input_text)

pos = use_regex(txt)[-1]
print(pos)
print(find_floating_point(pos))

