#!/usr/bin/env python3
"""
A utility script to clean Org mode syntax from files.

This script provides functionality to clean Org mode specific syntax elements
from .org files, making them more suitable for processing with LLMs or other
text analysis tools.

Example: Clean a single Org mode file:
BASH
python org_cleaner.py file input.org --output cleaned.txt

Clean all Org mode files in a directory:
BASH
python org_cleaner.py directory /path/to/org/files /path/to/output/directory

"""

import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain_community.document_loaders import TextLoader
from langchain.schema.document import Document


@dataclass
class OrgSyntaxPattern:
    """Dataclass to store regex patterns for Org mode syntax cleaning."""
    description: str
    pattern: str
    replacement: str = ''
    flags: int = re.DOTALL


class OrgModeLoader(TextLoader):
    """Custom loader for Org mode files with cleaning functionality."""

    # Define all patterns in one place for easier maintenance
    SYNTAX_PATTERNS = [
        OrgSyntaxPattern(
            description="Property drawers",
            pattern=r':PROPERTIES:\n(.+?)\n:END:',
            flags=re.DOTALL
        ),
        OrgSyntaxPattern(
            description="Logbook drawers",
            pattern=r':LOGBOOK:\n(.+?)\n:END:',
            flags=re.DOTALL
        ),
        OrgSyntaxPattern(
            description="Other drawers",
            pattern=r':(CLOCK|RESULTS|NOTES):\n(.+?)\n:END:',
            flags=re.DOTALL
        ),
        OrgSyntaxPattern(
            description="Org tags at end of headings",
            pattern=r'\s*:[A-Za-z0-9_@:]+:\s*$',
            flags=re.MULTILINE
        ),
        OrgSyntaxPattern(
            description="Org priorities",
            pattern=r'\s*\[#[A-Z]\]\s*',
            replacement=' '
        ),
        OrgSyntaxPattern(
            description="Org links with description",
            pattern=r'\[\[([^\]]+)\]\[([^\]]+)\]\]',
            replacement=r'\2'  # Keep description
        ),
        OrgSyntaxPattern(
            description="Org links without description",
            pattern=r'\[\[([^\]]+)\]\]',
            replacement=r'\1'  # Keep link target
        ),
        OrgSyntaxPattern(
            description="Org formatting markers",
            pattern=r'(\*\*|\/\/|==|~~|_)(.+?)\1',
            replacement=r'\2'  # Keep content
        ),
        OrgSyntaxPattern(
            description="Org comments",
            pattern=r'^\s*#\+.*$',
            flags=re.MULTILINE
        ),
        OrgSyntaxPattern(
            description="TODO/DONE states",
            pattern=r'^\s*\*+\s+(TODO|DONE|WAITING|CANCELLED|IN-PROGRESS)\s+',
            replacement=r'* ',
            flags=re.MULTILINE
        ),
        OrgSyntaxPattern(
            description="Active timestamps",
            pattern=r'<\d{4}-\d{2}-\d{2}(?: [A-Za-z]+)?(?: \d{2}:\d{2})?(?: [+-]\d+[dwmy])?(?: \d{2}:\d{2})?>(?:--<\d{4}-\d{2}-\d{2}(?: [A-Za-z]+)?(?: \d{2}:\d{2})?(?: [+-]\d+[dwmy])?(?: \d{2}:\d{2})?>)?'
        ),
        OrgSyntaxPattern(
            description="Inactive timestamps",
            pattern=r'\[\d{4}-\d{2}-\d{2}(?: [A-Za-z]+)?(?: \d{2}:\d{2})?(?: [+-]\d+[dwmy])?(?: \d{2}:\d{2})?\](?:--\[\d{4}-\d{2}-\d{2}(?: [A-Za-z]+)?(?: \d{2}:\d{2})?(?: [+-]\d+[dwmy])?(?: \d{2}:\d{2})?\])?'
        ),
        OrgSyntaxPattern(
            description="Checkbox markers",
            pattern=r'^\s*- \[([ X])\]\s*',
            replacement='- ',
            flags=re.MULTILINE
        ),
        OrgSyntaxPattern(
            description="Multiple blank lines",
            pattern=r'\n{3,}',
            replacement='\n\n'
        ),
    ]

    def __init__(self, file_path: str, encoding: Optional[str] = None, verbose: bool = False):
        """
        Initialize the OrgModeLoader.
        
        Args:
            file_path: Path to the Org mode file
            encoding: File encoding (default: None, uses system default)
            verbose: Whether to print detailed cleaning information
        """
        super().__init__(file_path, encoding=encoding)
        self.verbose = verbose

    def load(self) -> List[Document]:
        """Load and return documents from Org mode file with Org syntax cleaned."""
        try:
            with open(self.file_path, encoding=self.encoding) as f:
                text = f.read()
        except Exception as e:
            raise ValueError(f"Error reading file {self.file_path}: {str(e)}")

        # Clean Org mode specific syntax
        cleaned_text = self._clean_org_syntax(text)

        # Extract title from the file name
        file_name = os.path.basename(self.file_path)
        title = os.path.splitext(file_name)[0]

        # Print before and after to verify cleaning is working
        if self.verbose:
            print(f"Cleaning file: {file_name}")
            print(f"Original length: {len(text)}, Cleaned length: {len(cleaned_text)}")
            print(f"Removed {len(text) - len(cleaned_text)} characters ({((len(text) - len(cleaned_text)) / len(text) * 100):.1f}%)")

        metadata = {
            "source": self.file_path,
            "title": title,
            "original_length": len(text),
            "cleaned_length": len(cleaned_text)
        }

        return [Document(page_content=cleaned_text, metadata=metadata)]

    def _clean_org_syntax(self, text: str) -> str:
        """
        Remove Org mode specific syntax elements.
        
        Args:
            text: The Org mode text to clean
            
        Returns:
            Cleaned text with Org mode syntax removed
        """
        original_length = len(text)
        
        # Apply all patterns in sequence
        for pattern in self.SYNTAX_PATTERNS:
            before_length = len(text)
            text = re.sub(pattern.pattern, pattern.replacement, text, flags=pattern.flags)
            after_length = len(text)
            
            if self.verbose and before_length != after_length:
                chars_removed = before_length - after_length
                print(f"  - {pattern.description}: removed {chars_removed} characters")

        return text.strip()


def clean_org_text(text: str, verbose: bool = False) -> str:
    """
    Clean Org mode syntax from a text string.
    
    Args:
        text: String containing Org mode content
        verbose: Whether to print detailed cleaning information
        
    Returns:
        The cleaned text content
    """
    # Create a temporary instance just to use the cleaning method
    loader = OrgModeLoader("dummy_path", verbose=verbose)
    return loader._clean_org_syntax(text)


def clean_org_file(file_path: str, output_path: Optional[str] = None, 
                  encoding: Optional[str] = None, verbose: bool = False) -> str:
    """
    Clean an Org mode file and optionally save the result.
    
    Args:
        file_path: Path to the Org mode file to clean
        output_path: Path to save the cleaned content (if None, doesn't save)
        encoding: File encoding to use
        verbose: Whether to print detailed cleaning information
        
    Returns:
        The cleaned text content
    """
    loader = OrgModeLoader(file_path, encoding=encoding, verbose=verbose)
    documents = loader.load()
    cleaned_text = documents[0].page_content
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding=encoding or 'utf-8') as f:
            f.write(cleaned_text)
        if verbose:
            print(f"Cleaned content saved to: {output_path}")
    
    return cleaned_text


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean Org mode syntax from files")
    parser.add_argument("input", help="Input Org mode file or string")
    parser.add_argument("-o", "--output", help="Output file (if not provided, prints to stdout)")
    parser.add_argument("-e", "--encoding", help="File encoding")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed cleaning information")
    parser.add_argument("-s", "--string", action="store_true", help="Treat input as a string instead of a file path")
    
    args = parser.parse_args()
    
    if args.string:
        result = clean_org_text(args.input, verbose=args.verbose)
    else:
        result = clean_org_file(args.input, args.output, args.encoding, verbose=args.verbose)
    
    if not args.output:
        print(result)