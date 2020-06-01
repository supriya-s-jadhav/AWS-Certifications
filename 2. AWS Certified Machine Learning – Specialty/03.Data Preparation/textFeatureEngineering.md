# Data Preparation

Data preparation is one of the important part of data analysis and modelling which includes below things:

1. Categorical Encoding

Converting categorical values into numeric values using mappings and one-hot techniques.

2. Feature Engineering

Transformation features so they are ready for ML algorithms. Ensures the relevant features are used for the problem at hand.

3. Handling Missing Values

Removing incomplete, incorrect formatted, irrelevant or duplicated data.

<b> How to handle Formatting, Missing values, Duplicates, Invalid data and Encoding ? </b>

Tools used in AWS for data preparation :

1. SageMaker and Jupyter Notebooks (Adhoc way)

2. ETL jobs in AWS Glue (Reusable)

For any questions in exam about ETL job with data, think of AWS Glue.

## Categorical Encoding Examples

Nominal: color: {green, purple, blue}

Nominal: Evil: {true, false}

Ordinal: Size: {L > M > S}

Nominal: Order does not matter

Ordinal: Order does matter

It is important to know nominal and ordinal values, that is how we will encode them.

a) One-hot encoding

Example: In a house loan dataset, the house_type column has values: {House, Condo, apartment}. Coding them as {0,1,2} will make algorithm understand that House < Condo < apartment. Hence, one of the right way to encode it is: make three columns as below:

| Type | Type_condo | Type_house | Type_apartment |
|------|------------|------------|----------------|
| Condo |    1      |     0      |       0        |
| house |    0      |     1      |       0        |
| apartment | 0     |     0      |       1        |
| Condo |     1     |     0      |       0        |

b) Grouping

c) Mapping Rare Values

Categorical Encoding depends on:

1) ML algorithm you chose

2) Text into numbers

3) There is no "Golden Rule"

4) Many different Approaches

## Text Feature Engineering

Transforming text within our data so ML algorithms can better analyze it.

Splitting text into bite size pieces.

### N-Gram

Example: {She, is, giving, an, exam}

Unigram = 1 word token

bigram = 2 word token

trigram = 3 word token

She is giving an exam.

Unigram: N-gram, Size = 1

Bigram: N-gram, Size = 2
{She is, is giving, giving an, an exam}

### Orthogonal Sparse Bigram (OSB)

Creates groups of words of size 'n' and outputs every pair of words that includes the first word.

Creates groups of words that always include the first word.

Orthogonal (When two things are independent of each other)

Sparse (scattered or thinly distributed)

Bigram (2-gram or two words)

Example: {"he is a jedi anf he will save us}

OSB, size = 4

{"he_is","he__a","he___jedi"}
{"is_a","is__jedi","is___and"}
{"a_jedi","a__and","a___he"}
{"jedi_and","jedi__he","jedi___will"}
{"and_he","and__will","and___save"}
{"he_will","he__save","he___us"}
{"will_save","will_us"}
{"save_us"}

### Term Frequency - Inverse Document Frequency (tf-idf)

Represents how important a word or words are to a given set of text by providing appropriate weights to terms that are common and less common in the text.

Shows us the popularity of a words in text data by making common words like "the" or "and" less important.

Term Frequency : How frequent does a word appear

Inverse Document Frequency: Number of documents in which terms occur.

Inverse: Makes common words less meaningful.

Example:

### Text Feature Engineering Use cases

| Problem | Transformation | Why |
|---------|----------------|-----|
| Matching phrases in spam emails | N-Gram | Easier to compare whole phrases like "click here now" or "you're a winner" |
| Determining the subject matter of multiple PDFs | Tf-idf </br>Orthogonal Sparse Bigram | Filter less important words in PDFs.</br>Find common word combinations repeated throughout PDFs. |

### Other methods of transformation before doing N-gram or OSB

* Remove Punctuation

Sometimes removing punctuation is a good idea if we do not need them

* Lowercase Transformation

Using lowercase transformation can help standardize raw text.

* Cartesian Product Transformation

Creates a new feature from the combination of two or more text or categorical values.

Combining sets of words together.

Example:

| Book | Binding |
|------|---------|
| Software Engineering | SoftCover |

Removes Punctuation and applies Cartesian Product Transformation

| CPT |
|-----|
| Software_Softcover, Engineering_Softcover |

### Feature Engineering Dates

Translating dates into useful information


## Text Feature Engineering Summary

| Technique | Function |
|-----------|----------|
| N-Gram | Splits text by whitespace with window size n |
| Orthogonal Sparse Bigram (OSB) | Splits text by keeping first word and uses delimiter with remaining whitespaces between second word with window size n |
| Term Frequency - Inverse Document Frequency (tf-idf) | Helps us determine how important words are within multiple documents |
| Removing Punctuation | Removes punctuation from text |
| Lowercase | Lowercase all the words in the text |
| Cartesian Product | Combines words together to create new feature |
| Feature Engineering Dates | Extracts information from dates and creates new features |

