import re

def _next_key(lines):
    '''
    Extracts the string between the first pair of '<' and ' '
    or '<' and '>'.

    '''
    result = re.search('<([^!]+?)( |>)', lines)
    return result.group(1) if result else None


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
    return { k: v.strip(' ') for k, v in re.findall('([^ ]+?)="([^=]*)"', attrs)}


def _parse(content):
    '''
    Recursively parse the content.

    A UPF block has either one of the following formats:

        <KEY ATTR1="..." ATTR2="..." ...>DATA</KEY>     (a block with data)
        <KEY ATTR1="..." ATTR2="..." .../>              (a block without data)

    Each UPF block will be converted to a dict entry. If DATA contains no
    UPF block, the entry (key-value pair) will be

        KEY: {ATTR1: ..., ATTR2: ..., 'data': DATA}

    If DATA contains further UPF blocks, the 'data': DATA part will be replaced
    by the parsed UPF blocks, i.e.,
    
        KEY: {ATTR1: ..., ATTR2: ..., NESTED_KEY1: {...}, NESTED_KEY2: {...}, ...}

    '''
    block = {}

    while key := _next_key(content):
        if (result := re.search('<{0}([^>]*)>(.*)</{0}>'.format(key), content)) is None:
            result = re.search('<{0}([^>]*)(.*?)/>'.format(key), content)
        assert result.group(0).count('<') == result.group(0).count('>'), 'Unmatched < and >'

        attrs, data = result.group(1), result.group(2)
        block[key] = _attrs2dict(attrs) | (_parse(data) if _next_key(data) else {'data': data})
        content = content[result.end(0):]

    return block


def read_upf(fpath):
    with open(fpath, 'r') as f:
        content = f.read()
    content = content.replace('\n', ' ') # remove all newlines (for regex)
    return _parse(content)


############################################################
#                           Test
############################################################
import unittest

class TestUPFRead(unittest.TestCase):

    def test_next_key(self):
        teststr = '<UPF version="2.0.1">'
        self.assertEqual(_next_key(teststr), 'UPF')

        teststr = '<PP_INFO>'
        self.assertEqual(_next_key(teststr), 'PP_INFO')

    
    def test_attrs2dict(self):
        teststr = 'key1=" value1"  key2="value2 "  key3=" value3 "'
        self.assertEqual(_attrs2dict(teststr), {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'})


    def test_read_upf(self):
        upf = read_upf('./testfiles/In_ONCV_PBE-1.0.upf')
        self.assertEqual(upf['UPF']['version'], '2.0.1')
        self.assertEqual(upf['UPF']['PP_HEADER']['generated'], 'Generated using ONCVPSP code by D. R. Hamann')

if __name__ == '__main__':
    unittest.main()

