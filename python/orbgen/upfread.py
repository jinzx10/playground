import re

def _next_key(lines):
    i_angle = lines.find('<')
    i_space = lines[i_angle:].find(' ')
    return lines[i_angle+1 : i_angle + i_space] if i_angle >= 0 else ''

def _attrs2dict(attrs):
    '''
    Convert a string of attributes to a dictionary

    Parameter
    ---------
        attrs : str
            A string that contains one or more key-value pairs like
            'key1="value1"  key2="value2"  key3="value3"'.

    Return
    ------
        A dictionary of key-value pairs.

    '''
    return { k: v for k, v in re.findall('([^ ]+?)=" *([^= ]*) *"', attrs)}

def _parse(content):
    result = {}

    # Get the next key
    key = _next_key(content)
    idx = 0
    while (len(key) > 0):
        print(key)
        key, stride = _next_key(content[idx:])
        idx += stride

    result = re.search('<{0}(.*?)>(.*)</{0}>'.format(next_key), content)
    attrs = result.group(1) if result else ''
    print(attributes)
    data = result.group(2)

    return result


def read_upf(fpath):
    with open(fpath, 'r') as f:
        content = f.read()

    content = content.replace('\n', ' ')

    _parse(content)

    teststr='date="150105" comment="" element="In" pseudo_type="NC" relativistic="scalar" is_ultrasoft="F" '
    print(result)

    print(_attrs2dict(teststr))

############################################################
#                           Test
############################################################
import unittest
class TestUPFRead(unittest.TestCase):
    def test_read_upf(self):
        read_upf('./testfiles/In_ONCV_PBE-1.0.upf')


if __name__ == '__main__':
    unittest.main()
