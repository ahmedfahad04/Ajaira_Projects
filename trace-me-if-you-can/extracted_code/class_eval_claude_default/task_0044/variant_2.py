import re
import string
import gensim
from bs4 import BeautifulSoup
from functools import partial


class HtmlUtil:

    def __init__(self):
        self.SPACE_MARK = '-SPACE-'
        self.JSON_MARK = '-JSON-'
        self.MARKUP_LANGUAGE_MARK = '-MARKUP_LANGUAGE-'
        self.URL_MARK = '-URL-'
        self.NUMBER_MARK = '-NUMBER-'
        self.TRACE_MARK = '-TRACE-'
        self.COMMAND_MARK = '-COMMAND-'
        self.COMMENT_MARK = '-COMMENT-'
        self.CODE_MARK = '-CODE-'
        
        # Define processing pipeline
        self.html_processors = [
            self._mark_code_blocks,
            self._format_lists,
            self._format_paragraphs
        ]

    @staticmethod
    def __format_line_feed(text):
        return re.sub(re.compile(r'\n+'), '\n', text)

    def _mark_code_blocks(self, soup):
        """Mark code blocks with placeholder"""
        for tag in soup.find_all(name=['pre', 'blockquote']):
            tag.string = self.CODE_MARK

    def _format_lists(self, soup):
        """Format list items with proper bullet points and punctuation"""
        list_formatter = lambda item: self._format_list_item(item)
        
        for container in soup.find_all(name=['ul', 'ol']):
            list(map(list_formatter, container.find_all('li')))

    def _format_list_item(self, li_item):
        """Format individual list item"""
        text = li_item.get_text().strip()
        if text:
            ending = '' if text.endswith(tuple(string.punctuation)) else '.'
            li_item.string = f'[-]{text}{ending}'

    def _format_paragraphs(self, soup):
        """Format paragraphs with context-sensitive punctuation"""
        paragraph_formatter = lambda p: self._format_paragraph_item(p)
        list(map(paragraph_formatter, soup.find_all(name=['p'])))

    def _format_paragraph_item(self, p_item):
        """Format individual paragraph"""
        text = p_item.get_text().strip()
        if text:
            if text[-1] not in string.punctuation:
                next_elem = p_item.find_next_sibling()
                has_code_sibling = next_elem and self.CODE_MARK in next_elem.get_text()
                suffix = ':' if has_code_sibling else '.'
                p_item.string = text + suffix
            else:
                p_item.string = text

    def format_line_html_text(self, html_text):
        if html_text is None or len(html_text) == 0:
            return ''
            
        soup = BeautifulSoup(html_text, 'lxml')
        
        # Apply processing pipeline
        for processor in self.html_processors:
            processor(soup)

        clean_text = gensim.utils.decode_htmlentities(soup.get_text())
        return self.__format_line_feed(clean_text)

    def extract_code_from_html_text(self, html_text):
        formatted_text = self.format_line_html_text(html_text)

        if self.CODE_MARK not in formatted_text:
            return []

        soup = BeautifulSoup(html_text, 'lxml')
        code_tags = soup.find_all(name=['pre', 'blockquote'])
        expected_count = formatted_text.count(self.CODE_MARK)
        
        code_extractor = lambda tag: tag.get_text()
        return list(filter(None, map(code_extractor, code_tags[:expected_count])))
