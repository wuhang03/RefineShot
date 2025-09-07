import pandas as pd
import json
import ast
import random
import sys

def load_category_mapping():
    """加载类别映射JSON文件"""
    try:
        with open('category.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: category.json not found")
        return {}
    except Exception as e:
        print(f"Error while loading category.json: {e}")
        return {}

def find_subcategory(answer_text, category_mapping, main_category):
    """Find the subcategory based on answer text and main category"""
    if main_category not in category_mapping:
        return None, []
    
    breakdown = category_mapping[main_category].get('breakdown', {})
    for subcategory, items in breakdown.items():
        if answer_text in items:
            return subcategory, items
    
    return None, []

def create_shuffled_options(items, correct_item):
    """
    Create shuffled options from items and return (options_json_str, answer_letter).
    `correct_item` should match exactly one item in `items`.
    """
    choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    shuffled_items = items[:]  # copy
    random.shuffle(shuffled_items)
    
    options_dict = {}
    answer_letter = None
    
    for i, item in enumerate(shuffled_items):
        if i >= len(choice_letters):
            break
        letter = choice_letters[i]
        options_dict[letter] = item
        if item == correct_item:
            answer_letter = letter
    
    return json.dumps(options_dict, ensure_ascii=False), answer_letter

def extract_answers_from_tsv():
    try:
        category_mapping = load_category_mapping()
        df = pd.read_csv('evaluation/data/ShotBench/test_origin.tsv', sep='\t', encoding='utf-8')

        # require 'question' column too since we may modify it for binary cases
        for col in ['answer', 'options', 'category', 'question']:
            if col not in df.columns:
                print(f"Error: Missing column '{col}' in file")
                return

        df_new = df.copy()
        rows_to_keep = []
        binary_count = 0
        multi_count = 0
        skipped_count = 0  # Counter for skipped rows

        for index, row in df.iterrows():
            answer_key = str(row['answer']).strip()
            options_str = str(row['options'])
            category = str(row['category']).strip()

            if category not in ["lighting type", "lighting", "shot framing"]:
                # Instead of skipping, directly copy the row to the new file
                rows_to_keep.append(index)
                # skipped_count += 1  # Increment skipped counter
                continue

            try:
                options_str = options_str.replace('""', '"')
                try:
                    options_dict = ast.literal_eval(options_str)
                except:
                    options_dict = json.loads(options_str)

                if answer_key in options_dict:
                    original_answer_text = options_dict[answer_key]
                    if ',' in original_answer_text:
                        print(f"Multi-answer detected (row {index+1}): {original_answer_text}")
                        rows_to_keep.append(index)
                        # skipped_count += 1  # Increment skipped counter
                        continue

                    subcategory, subcategory_items = find_subcategory(original_answer_text, category_mapping, category)
                    print("original_answer_text:", original_answer_text)  # Debug output
                    print("subcategory:", subcategory)  # Debug output
                    print("subcategory_items:", subcategory_items)  # Debug output

                    if subcategory_items:
                        if len(subcategory_items) == 1:
                            # Binary question: create yes/no options, then shuffle and update answer
                            new_question = f"Is the {category} of this shot {original_answer_text}?"
                            yes_no_items = ["yes", "no"]
                            # correct answer is "yes" (since original_answer_text matches the single subcategory item)
                            new_options, new_answer = create_shuffled_options(yes_no_items, "yes")

                            if new_answer is None:
                                print(f"Error: after shuffling binary options, correct answer not found (row {index+1})")
                                # skipped_count += 1  # Increment skipped counter
                                continue

                            df_new.at[index, 'question'] = new_question
                            df_new.at[index, 'options'] = new_options
                            df_new.at[index, 'answer'] = new_answer
                            rows_to_keep.append(index)
                            binary_count += 1

                            # debug output (English)
                            print(f"Row {index+1} (Binary):")
                            print(f"  Question: {new_question}")
                            print(f"  Answer key: {answer_key} -> {new_answer}")
                            print(f"  Answer text (original): {original_answer_text}")
                            print(f"  Main category: {category}")
                            print(f"  Subcategory: {subcategory}")
                            print(f"  Original options: {options_str}")
                            print(f"  New options: {new_options}")
                            print("-" * 60)
                        else:
                            # Multi-choice: shuffle subcategory items and update answer letter
                            new_options, new_answer = create_shuffled_options(subcategory_items, original_answer_text)
                            if new_answer is None:
                                print(f"Answer not found in new options (row {index+1}): {original_answer_text}")
                                # skipped_count += 1  # Increment skipped counter
                                continue

                            df_new.at[index, 'options'] = new_options
                            df_new.at[index, 'answer'] = new_answer
                            rows_to_keep.append(index)
                            multi_count += 1

                            # debug output (English)
                            print(f"Row {index+1}:")
                            print(f"  Answer key: {answer_key} -> {new_answer}")
                            print(f"  Answer text: {original_answer_text}")
                            print(f"  Main category: {category}")
                            print(f"  Subcategory: {subcategory}")
                            print(f"  Original options: {options_str}")
                            print(f"  New options: {new_options}")
                            print("-" * 60)
                    else:
                        sys.exit(0)
                        print(f"Subcategory not found (row {index+1}): {original_answer_text}")
                        skipped_count += 1  # Increment skipped counter
                        sleep(10)
                else:
                    print(f"Warning: answer key '{answer_key}' not found in options (row {index+1})")
                    # skipped_count += 1  # Increment skipped counter

            except Exception as e:
                print(f"Error parsing options (row {index+1}): {e}")
                print(f"Original options: {options_str}")
                # skipped_count += 1  # Increment skipped counter

        df_result = df_new.loc[rows_to_keep]
        df_result.to_csv('evaluation/data/ShotBench/test_replace.tsv', sep='\t', index=False, encoding='utf-8')

        print(f"\nProcessing complete: {len(df)} rows processed, {len(df_result)} rows saved to test_replace.tsv")
        print(f"  Binary (yes/no) converted: {binary_count}")
        print(f"  Multi-choice converted: {multi_count}")
        print(f"  Rows skipped: {skipped_count}")  # Output skipped count

    except FileNotFoundError:
        print("Error: test_origin.tsv not found")
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    extract_answers_from_tsv()