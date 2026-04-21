import re
import string
import gensim
from bs4 import BeautifulSoup


class HtmlTextProcessor:
    """Encapsulates HTML text processing logic"""
    
    def __init__(self, code_mark):
        self.code_mark = code_mark
    
    def process_code_elements(self, soup):
        code_elements = soup.find_all(name=['pre', 'blockquote'])
        for element in code_elements:
            element.string = self.code_mark
        return code_elements
    
    def process_list_elements(self, soup):
        for list_element in soup.find_all(name=['ul', 'ol']):
            for list_item in list_element.find_all('li'):
                self._format_list_item(list_item)
    
    def _format_list_item(self, item):
        content = item.get_text().strip()
        if content:
            needs_period = content[-1] not in string.punctuation
            formatted_content = f'[-]{content}{"." if needs_period else ""}'
            item.string = formatted_content
    
    def process_paragraph_elements(self, soup):
        for paragraph in soup.find_all(name=['p']):
            self._format_paragraph(paragraph)
    
    def _format_paragraph(self, paragraph):
        content = paragraph.get_text().strip()
        if content:
            if content[-1] in string.punctuation:
                paragraph.string = content
            else:
                next_element = paragraph.find_next_sibling()
                is_followed_by_code = (next_element and 
                                     self.code_mark in next_element.get_text())
                suffix = ':' if is_followed_by_code else '.'
                paragraph.string = content + suffix


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

    @staticmethod
    def __format_line_feed(text):
        return re.sub(re.compile(r'\n+'), '\n', text)

    def format_line_html_text(self, html_text):
        if html_text is None or len(html_text) == 0:
            return ''
        
        soup = BeautifulSoup(html_text, 'lxml')
        processor = HtmlTextProcessor(self.CODE_MARK)
        
        processor.process_code_elements(soup)
        processor.process_list_elements(soup)
        processor.process_paragraph_elements(soup)

        clean_text = gensim.utils.decode_htmlentities(soup.get_text())
        return self.__format_line_feed(clean_text)

    def extract_code_from_html_text(self, html_text):
        processed_text = self.format_line_html_text(html_text)

        if self.CODE_MARK not in processed_text:
            return []

        soup = BeautifulSoup(html_text, 'lxml')
        processor = HtmlTextProcessor(self.CODE_MARK)
        code_elements = processor.process_code_elements(soup)
        
        expected_code_blocks = processed_text.count(self.CODE_MARK)
        extracted_codes = []
        
        for index in range(min(len(code_elements), expected_code_blocks)):
            code_content = code_elements[index].get_text()
            if code_content:
                extracted_codes.append(code_content)
                
        return extracted_codes
