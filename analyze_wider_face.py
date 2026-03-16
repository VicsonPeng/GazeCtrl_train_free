import os
from collections import defaultdict

def analyze_dataset(directory):
    total_images = 0
    categories = []
    
    # Iterate through all category folders
    for category_name in sorted(os.listdir(directory)):
        category_path = os.path.join(directory, category_name)
        
        if os.path.isdir(category_path):
            images = [f for f in os.listdir(category_path) if f.casefold().endswith(('.jpg', '.jpeg', '.png'))]
            count = len(images)
            categories.append({"name": category_name, "count": count})
            total_images += count

    # Sort categories by image count descending
    categories.sort(key=lambda x: x['count'], reverse=True)

    # Generate Markdown Report
    report_content = f"""# WIDER Face Dataset Analysis Report

## Overview
- **Dataset Path:** `{directory}`
- **Total Categories:** {len(categories)}
- **Total Images:** {total_images}

## Category Breakdown (Sorted by Image Count)

| Category Name | Image Count | Percentage |
| :--- | :---: | :---: |
"""
    for cat in categories:
        percentage = (cat['count'] / total_images) * 100 if total_images > 0 else 0
        report_content += f"| {cat['name']} | {cat['count']} | {percentage:.2f}% |\n"

    # Save Report
    report_path = os.path.join(os.path.dirname(__file__), "wider_face_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"Analysis complete. Report saved to: {report_path}")

if __name__ == "__main__":
    dataset_path = r"C:\Users\vicso\Codes\cplab_gaze_control\dataset\wider_face\WIDER_train\images"
    analyze_dataset(dataset_path)
