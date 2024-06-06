"""
    Example review from the Stanford dataset
    and the result of the same text after cleaning.
"""
from utils_data_processing import clean_text

text = """Is it worth seeing? If you can look past the technical and budgetary limitations, and get into the story, I think you
will enjoy this, especially if you've actually read the original H G Wells novel. 
If, however, you are easily put off by cheap production values, you'd best pass on this (unless you're a MST3K fan).
Be warned, however that the film runs a full 3 hours, so I don't recommend watching it all in one sitting.<br /><br />
BTW: An entirely different version of War of the Worlds (aka "INVASION") came out on DVD the same month that
Spielberg's hit the theatres: http://imdb.com/title/tt0449040/.
This was also made on a budget, but is updated to the present day like the Spielberg film - but it's much better! And to top it off, Jeff Wayne is making an animated film of his best-selling album from 1978, but that won't be out until 2007."""

if __name__ == "__main__":
    print("---\nraw review\n---")
    print(text)
    print("\n---\ncleaned review\n---")
    print(clean_text(text))
