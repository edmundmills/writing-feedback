from utils.text import *

def test_to_sentences():
    text = "Good morning Dr. Adams. The patient is waiting for you in room number 3."
    sentences = ['Good morning Dr. Adams.', 'The patient is waiting for you in room number 3.']
    assert(to_sentences(text) == sentences)