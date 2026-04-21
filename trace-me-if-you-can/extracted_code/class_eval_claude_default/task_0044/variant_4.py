import re
import string
import gensim
from bs4 import BeautifulSoup
from enum import Enum


class MarkType(Enum):
    SPACE = '-SPACE-'
    JSON = '-JSON-'
    MARKUP_LANGUAGE = '-MARKUP_LANGUAGE-'
    URL = '-URL-'
    NUMBER = '-NUMBER-'
    TRACE = '-TRACE-'
    COMMAND = '-COMMAND-'
    COMMENT = '-COMMENT-'
    CODE = '-CODE-'


class HtmlUtil:

    def __init__(self):
        self.SPACE_MARK = MarkType.SPACE.value
        self.JSON_MARK = MarkType.JSON.value
        self.MARKUP_LANGUAGE_MARK = MarkType.MARKUP_LANGUAGE.value
        self.URL_MARK = MarkType.URL.value
        self.NUMBER_MARK = MarkType.NUMBER.value
        self.TRACE_MARK = MarkType.TRACE.value
        self.COMMAND_MARK = MarkType.COMMAND.value
        self.COMMENT_MARK = MarkType.COMMENT.value
        self.CODE_MARK = MarkType.CODE.value

    @staticmethod
    def __format_line_feed(text):
        return re.sub(re.compile(r'\n+'), '\n', text)

    def _apply_text_formatting(self, element, text, context=None):
        """Apply formatting rules based on element type and context"""
        if not text:
            return
            
        formatting_rules = {
            'needs_period': text[-1] not in string.punctuation,
            'has_code_sibling': (context and context.get('next_sibling') and 
                               self.CODE_MARK in context['next_sibling'].get_text()),
            'is_list_item': context and context.get('is_list_item', False)
        }
        
        if formatting_rules['is_list_item']:
            suffix = '.' if formatting_rules['needs_period'] else ''
            element.string = f'[-]{text}{suffix}'
        elif formatting_rules['needs_period']:
            suffix = ':' if formatting_rules['has_code_sibling'] else '.'
            element.string = text + suffix
        else:
            element.string = text

    def format_line_html_text(self, html_text):
        if html_text is None or len(html_text) == 0:
            return ''
        soup = BeautifulSoup(html_text, 'lxml')

        # Process code blocks
        code_elements = soup.find_all(name=['pre', 'blockquote'])
        for element in code_elements:
            element.string = self.CODE_MARK

        # Process lists with context
        for list_container in soup.find_all(name=['ul', 'ol']):
            for list_item in list_container.find_all('li'):
                item_text = list_item.get_text().strip()
                if item_text:
                    context = {'is_list_item': True}
                    self._apply_text_formatting(list_item, item_text, context)

        # Process paragraphs with sibling context
        for paragraph in soup.find_all(name=['p']):
            paragraph_text = paragraph.get_text().strip()
            if paragraph_text:
                context = {'next_sibling': paragraph.find_next_sibling()}
                self._apply_text_formatting(paragraph, paragraph_text, context)

        clean_text = gensim.utils.decode_htmlentities(soup.get_text())
        return self.__format_line_feed(clean_text)

    def extract_code_from_html_text(self, html_text):
        formatted_text = self.format_line_html_text(html_text)

        code_mark_count = formatted_text.count(self.CODE_MARK)
        if code_mark_count == 0:
            return []

        soup = BeautifulSoup(html_text, 'lxml')
        code_elements = soup.find_all(name=['pre', 'blockquote'])
        
        result = []
        for i in range(min(code_mark_count, len(code_elements))):
            code_text = code_elements[i].get_text()
            if code_text:
                result.append(code_text)
        return result
