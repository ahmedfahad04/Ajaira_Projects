import re
import string
from bs4 import BeautifulSoup
import gensim.utils


class HtmlSanitizer:

    def __init__(self):
        self.markers = {
            'space': '-SPACE-',
            'json': '-JSON-',
            'markup': '-MARKUP_LANGUAGE-',
            'url': '-URL-',
            'number': '-NUMBER-',
            'trace': '-TRACE-',
            'command': '-COMMAND-',
            'comment': '-COMMENT-',
            'code': '-CODE-'
        }

    @staticmethod
    def __fix_newlines(text):
        return re.sub(r'\n+', '\n', text)

    def sanitize_html(self, html_text):
        if not html_text:
            return ''
        soup = BeautifulSoup(html_text, 'lxml')

        for tag in soup.find_all(['pre', 'blockquote']):
            tag.string = self.markers['code']

        for list_tag in soup.find_all(['ul', 'ol']):
            for li in list_tag.find_all('li'):
                li_text = li.get_text().strip()
                if li_text:
                    if li_text[-1] in string.punctuation:
                        li.string = f'- {li_text}'
                    else:
                        li.string = f'- {li_text}.'

        for p in soup.find_all('p'):
            p_text = p.get_text().strip()
            if p_text:
                if p_text[-1] in string.punctuation:
                    p.string = p_text
                elif p.find_next_sibling() and self.markers['code'] in p.find_next_sibling().get_text():
                    p.string = f'{p_text}:'
                else:
                    p.string = f'{p_text}.'

        clean_text = gensim.utils.decode_htmlentities(soup.get_text())
        return self.__fix_newlines(clean_text)

    def isolate_code_from_html(self, html_text):
        sanitized_text = self.sanitize_html(html_text)

        if self.markers['code'] not in sanitized_text:
            return []

        code_indexes = [i for i, mark in enumerate(sanitized_text.split(self.markers['code'])) if mark]
        soup = BeautifulSoup(html_text, 'lxml')
        code_tags = soup.find_all(['pre', 'blockquote'])
        return [code_tags[index].get_text() for index in code_indexes]
