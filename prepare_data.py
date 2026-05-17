import pandas as pd

fake_df = pd.read_csv("fake.csv")
true_df = pd.read_csv("true.csv")

fake_df['label'] = 0
true_df['label'] = 1


min_len=min(len(fake_df),len(true_df))
fake_df = fake_df.sample(min_len,random_state=42)
true_df = true_df.sample(min_len,random_state=42)

combined_df = pd.concat([fake_df, true_df], axis=0)

print(combined_df['label'].value_counts())
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
combined_df['content'] = combined_df['title'] + ' ' + combined_df['text']


combined_df = combined_df[['content','label']]
combined_df.to_csv("fake_train.csv",index=False)


print("✅ fake_train.csv created successfully.")
