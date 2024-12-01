# Association_rules
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# List of transactions
trans = [
    ['Eggs','Milk','Bread'],
    ['Eggs','Apple'],
    ['Milk','Bread'],
    ['Apple','Milk'],
    ['Milk','Apple','Bread']
]

# Transform transactions to one-hot encoded DataFrame
tc = TransactionEncoder()
tc_array = tc.fit(trans).transform(trans)
df = pd.DataFrame(tc_array, columns=tc.columns_)
print("Encoded DataFrame:")
print(df)

# Generate frequent item sets
freq = apriori(df, min_support=0.5, use_colnames=True)
print("\nFrequent Item sets:")
print(freq)

# Generate association rules
rules = association_rules(freq, metric='support', min_threshold=0.05)

# Sort rules by support and confidence in descending order
rules_sorted = rules.sort_values(['support', 'confidence'], ascending=[False, False])
print("\nAssociation Rules:")
print(rules_sorted)