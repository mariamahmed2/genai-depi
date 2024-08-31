## Notes on CounterVectorizer & TF-IDF Vectorizer
`max_df` is used for removing terms that appear too frequently, also known as "corpus-specific stop words". 
* `max_df`: `float` = 0.50 means "ignore terms that appear in more than 50% of the documents".
* `max_df`: `int` = 25 means "ignore terms that appear in more than 25 documents".
* The default `max_df` is `1.0`, which means "ignore terms that appear in more than 100% of the documents". Thus, the default setting does not ignore any terms.
------------
`min_df` is used for removing terms that appear too infrequently. For example:

* `min_df`: `float` = 0.01 means "ignore terms that appear in less than 1% of the documents".
* `min_df`: `int` = 5 means "ignore terms that appear in less than 5 documents".
* The default `min_df` is `1`, which means "ignore terms that appear in less than 1 document". Thus, the default setting does not ignore any terms.
------------