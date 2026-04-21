import re
import string
import gensim
from bs4 import BeautifulSoup


class HtmlUtil:

    def __init__(self):
        self.markers = {
            'SPACE': '-SPACE-',
            'JSON': '-JSON-',
            'MARKUP_LANGUAGE': '-MARKUP_LANGUAGE-',
            'URL': '-URL-',
            'NUMBER': '-NUMBER-',
            'TRACE': '-TRACE-',
            'COMMAND': '-COMMAND-',
            'COMMENT': '-COMMENT-',
            'CODE': '-CODE-'
        }

    @staticmethod
    def __format_line_feed(text):
        return re.sub(r'\n+', '\n', text)

    def _process_code_tags(self, soup):
        """Extract and mark code tags"""
        code_tags = soup.find_all(name=['pre', 'blockquote'])
        for tag in code_tags:
            tag.string = self.markers['CODE']
        return code_tags

    def _process_list_items(self, soup):
        """Process ul/ol list items with proper punctuation"""
        for list_container in soup.find_all(name=['ul', 'ol']):
            for li_item in list_container.find_all('li'):
                text = li_item.get_text().strip()
                if text:
                    punctuation_suffix = '' if text[-1] in string.punctuation else '.'
                    li_item.string = f'[-]{text}{punctuation_suffix}'

    def _process_paragraphs(self, soup):
        """Process paragraph tags with context-aware punctuation"""
        for p_item in soup.find_all(name=['p']):
            text = p_item.get_text().strip()
            if text:
                if text[-1] not in string.punctuation:
                    next_elem = p_item.find_next_sibling()
                    suffix = ':' if (next_elem and self.markers['CODE'] in next_elem.get_text()) else '.'
                    p_item.string = text + suffix
                else:
                    p_item.string = text

    def format_line_html_text(self, html_text):
        if not html_text:
            return ''
        
        soup = BeautifulSoup(html_text, 'lxml')
        self._process_code_tags(soup)
        self._process_list_items(soup)
        self._process_paragraphs(soup)
        
        clean_text = gensim.utils.decode_htmlentities(soup.get_text())
        return self.__format_line_feed(clean_text)

    def extract_code_from_html_text(self, html_text):
        formatted_text = self.format_line_html_text(html_text)
        
        if self.markers['CODE'] not in formatted_text:
            return []

        soup = BeautifulSoup(html_text, 'lxml')
        code_tags = soup.find_all(name=['pre', 'blockquote'])
        code_count = formatted_text.count(self.markers['CODE'])
        
        return [tag.get_text() for tag in code_tags[:code_count] if tag.get_text()]
