# ChronoBerg: A Dataset for Temporal Analysis of Hateful Content and Machine Unlearning
---

## File and Directory Descriptions

### **[`Constructing a Chronological Dataset from 250 Years of Literature/`](Constructing%20a%20Chronological%20Dataset%20from%20250%20Years%20of%20Literature/)**
- **Chapter 3**
- **Files**:
  - **[`Hate Analysis/`](Constructing%20a%20Chronological%20Dataset%20from%20250%20Years%20of%20Literature/Hate%20Analysis/)**
    - [`analyze_chronoberg_with_fb_roberta_hs_model.py`](Constructing%20a%20Chronological%20Dataset%20from%20250%20Years%20of%20Literature/Hate%20Analysis/analyze_chronoberg_with_fb_roberta_hs_model.py): Analyze dataset for hateful sentences using [Facebook RoBERTa Hate Speech model](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target)
    - [`analyze_chronoberg_with_perspective_api.ipynb`](Constructing%20a%20Chronological%20Dataset%20from%20250%20Years%20of%20Literature/Hate%20Analysis/analyze_chronoberg_with_perspective_api.ipynb): Analyze a set of potentially hateful sentences using [Perspective API](https://perspectiveapi.com/). Also includes the construction of sets of hateful sentences for different thresholds and the sampling of 100 random sentences as discussed in the paper.
    - [`requirements_fb_roberta_analysis.txt`](Constructing%20a%20Chronological%20Dataset%20from%20250%20Years%20of%20Literature/Hate%20Analysis/requirements_fb_roberta_analysis.txt): Libraries required to run `analyze_chronoberg_with_fb_roberta_hs_model.py`
  - [`Build extended PG catalogue.ipynb`](Constructing%20a%20Chronological%20Dataset%20from%20250%20Years%20of%20Literature/Build%20extended%20PG%20catalogue.ipynb): Jupyter notebook to build an extended Project Gutenberg catalogue given the book's RDF files have been downloaded from a [mirror](https://www.gutenberg.org/MIRRORS.ALL)
  - [`ChronoBerg_Statistics.ipynb`](Constructing%20a%20Chronological%20Dataset%20from%20250%20Years%20of%20Literature/ChronoBerg_Statistics.ipynb): Jupyter notebook to reproduce figures and tables from the paper
  - [`Fetching_years_in_which_books_have_been_created.ipynb`](Constructing%20a%20Chronological%20Dataset%20from%20250%20Years%20of%20Literature/Fetching_years_in_which_books_have_been_created.ipynb): Jupyter notebook to infer a book's publication year given its title and its author from different sources such as [Open Library API](https://openlibrary.org/dev/docs/api/search) and [Wikipedia](https://www.mediawiki.org/wiki/API:Main_page)
  - [`Group_books_by_year.ipynb`](Constructing%20a%20Chronological%20Dataset%20from%20250%20Years%20of%20Literature/Group_books_by_year.ipynb): Constructs one `.txt` file per year covered in the dataset by concatenating all books from the given year if `.txt` files for the books are available. Assumes all `.txt` book files have been mirrored. ([Learn more about mirroring](https://www.gutenberg.org/help/mirroring.html))
  - [`Polish text files for fb_roberta.ipynb`](Constructing%20a%20Chronological%20Dataset%20from%20250%20Years%20of%20Literature/Polish%20text%20files%20for%20fb_roberta.ipynb): Prepares files for Facebook RoBERTa Hate Speech analysis by removing trailing whitespaces and quotation marks (`"`)
  - [`build_retain_sets.ipynb`](Constructing%20a%20Chronological%20Dataset%20from%20250%20Years%20of%20Literature/build_retain_sets.ipynb): Jupyter notebook for building a retain set given the dataset and a forget set (set of hateful sentences)
  - [`word analyses.ipynb`](Constructing%20a%20Chronological%20Dataset%20from%20250%20Years%20of%20Literature/word%20analyses.ipynb): Jupyter notebook to conduct analyses on the ChronoBerg dataset regarding word frequencies

### **[`Evaluating Models for Hate Speech Detection/`](Evaluating%20Models%20for%20Hate%20Speech%20Detection/)**
- **Chapter 2**
- **Files**:
  - [`Evaluate_different_HS_models.ipynb`](Evaluating%20Models%20for%20Hate%20Speech%20Detection/Evaluate_different_HS_models.ipynb): Jupyter notebook for testing nine popular hate speech detection models on the [HateCheck benchmark](https://github.com/paul-rottger/hatecheck-data) 

### **[`Unlearning Hateful Content from ChronoBerg/`](Unlearning%20Hateful%20Content%20from%20ChronoBerg/)**
- **Chapter 4**
- **Files**:
  - [`LLM_hate_test.ipynb`](Unlearning%20Hateful%20Content%20from%20ChronoBerg/LLM_hate_test.ipynb): Jupyter notebook for testing hate in query completion of Llama, Llama Chronoberg, Llama Instruct, and Aurora-M

---
