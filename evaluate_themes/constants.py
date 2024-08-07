system_prompts = {
    "physical_appearance": """You are tasked with analyzing a given text to determine if it includes any information about the physical appearance of a person. Your output should be a JSON object with two keys:

1. `contains_appearance_info`: a boolean indicating whether the text contains information about physical appearance.
2. `appearance_info`: An array of strings describing the physical appearance issues mentioned in the text. If no such details are mentioned, an empty array.

Examples:

Input: "Mr Smith is a 95 year old man who lives on his own. When I visited, his clothes were dirty and scruffy. He reports he does not need support. He has anemia, arthritis and impaired mobility and requires support to move around the home."
Output: {
  "contains_appearance_info": true,
  "appearance_info": ["dirty", "scruffy"]
}

Input: "Mrs Smith is a 89 year old woman. She has dementia, a history of wandering, and lacks insight into her care needs. She has a grade 2 pressure ulcer and depends on carers to transfer in and out of bed. She appears disshevelled. She is incontinent of urine."
Output: {
  "contains_appearance_info": true,
  "appearance_info": ["disshevelled"]
}

Input: "Mr Smith lives alone in a council flat. He enjoys spending time with his grandchildren and loves reading historical novels. He attends a local book club every week and participates actively in discussions.
Output: {
  "contains_appearance_info": false,
  "appearance_info": []
}


Input: "Mrs Smith is a 65 year old woman. She is physically well but has advanced dementia.  She tends to sleep a lot during the day, becoming aggressive and confused when encouraged to keep awake."
Output: {
  "contains_appearance_info": false,
  "appearance_info": []
}

Please read the provided text carefully and generate the appropriate JSON output based on your findings. Please ensure the output is valid JSON.
""",
    "health": """You are tasked with analyzing a given text to determine if it includes any information about the physical and mental health of a person. Your output should be a JSON object with four keys:

1. `contains_physical_health_info`: a boolean indicating whether the text contains information about physical health.
2. `physical_health_details`: An array of strings describing the physical health issues mentioned in the text. If no such details are mentioned, an empty array.
1. `contains_mental_health_info`: a boolean indicating whether the text contains information about mental health.
2. `mental_health_details`: An array of strings describing the mental health issues mentioned in the text. If no such details are mentioned, an empty array.

An individual may have multiple physical and mental health conditions. However, a health condition may only be a physical or mental health issue, but not both. Examples:


Input: "Mr Smith is a 95 year old man who lives on his own. When I visited, his clothes were dirty and scruffy. He reports he does not need support. He has anemia, arthritis and impaired mobility and requires support to move around the home."
Output: {
  "contains_physical_health_info": true,
  "physical_health_details": ["anemia", "arthritis", "impaired mobility", "support to move around the home"],
  "contains_mental_health_info" : false,
  "mental_health_details" : []
}

Input: "Mrs Smith is a 89 year old woman. She has dementia, a history of wandering, and lacks insight into her care needs. She has a grade 2 pressure ulcer and depends on carers to transfer in and out of bed. She is incontinent of urine."

Output: {
  "contains_physical_health_info": true,
  "physical_health_details": ["grade 2 pressure ulcer", "depends on carers to transfer in and out of bed", "incontinence"],
  "contains_mental_health_info" : true,
  "mental_health_details" : ["dementia", "history of wandering", "lacks insight into care needs"]
}

Input: "Mr Smith lives alone in a council flat. He enjoys spending time with his grandchildren and loves reading historical novels. He attends a local book club every week and participates actively in discussions.
Output: {
  "contains_physical_health_info": false,
  "physical_health_details": [],
  "contains_mental_health_info" : false,
  "mental_health_details" : []
}

Input: "Mrs Smith is a 65 year old woman. She is physically well but has advanced dementia.  She tends to sleep a lot during the day, becoming aggressive and confused when encouraged to keep awake."
Output: {
  "contains_physical_health_info": false,
  "physical_health_details": [],
  "contains_mental_health_info" : true,
  "mental_health_details" : ["advanced dementia", "aggression", "confusion"]
}

Please read the provided text carefully and generate the appropriate JSON output based on your findings. Please ensure the output is valid JSON.
""",
    "subjective_language": """You are tasked with analyzing a given text to determine if it includes subjective or judgmental language about a person. By this I mean the language includes a judgment from the perspective of the author of the text, about the person whom the text describes. Your output should be a JSON object with two keys:

1. `contains_subjective_language`: a boolean indicating whether the text contains subjective or judgmental language.
2. `subjective_language_info`: An array of strings describing the language used in the text. If no such details are mentioned, an empty array.

Examples:

Input: "Smith is a 95 year old man who lives on his own. When I visited, his clothes were dirty and scruffy. He reports he does not need support. He has anemia, arthritis and impaired mobility and requires support to move around the home."
Output: {
  "contains_subjective_language": true,
  "subjective_language_info": ["dirty", "scruffy"]
}

Input: "Mrs Smith is a 89 year old woman. She has dementia, a history of wandering, and lacks insight into her care needs. She has a grade 2 pressure ulcer and depends on carers to transfer in and out of bed. She appears disshevelled. She is incontinent of urine."
Output: {
  "contains_subjective_language": true,
  "subjective_language_info": ["lacks insight", "disshevelled"]
}

Input: "Mr Smith lives alone in a council flat. He spends time with his grandchildren and reads historical novels. He attends a local book club every week.
Output: {
  "contains_subjective_language": false,
  "subjective_language_info": []
}


Input: "Mr Smith is very vocal and has repeatedly stated that he is capable of supporting himself and requires no support from others."
Output: {
  "contains_subjective_language": true,
  "subjective_language_info": ["vocal"]
}

Input: "Mrs Smith is a 65 year old woman. She declines care. She makes unwise decisions about her care."
Output: {
  "contains_subjective_language": true,
  "subjective_language_info": ["unwise"]
}

Input: "Mrs Smith is a 70 years old woman who lives alone in a flat situated on the london, accessible by lift. She has three children. She reports her favourite food is egg salad."
Output: {
  "contains_subjective_language": false,
  "subjective_language_info": []
}

Input: "Mrs Smith is a 65 year old woman. She declines care. She makes unwise decisions about her care."
Output: {
  "contains_subjective_language": true,
  "subjective_language_info": ["unwise"]
}

Please read the provided text carefully and generate the appropriate JSON output based on your findings. Please ensure the output is valid JSON.
""",
}

system_prompt_keys = {
    "health": {
        "contains_physical_health_info",
        "physical_health_details",
        "contains_mental_health_info",
        "mental_health_details",
    },
    "physical_appearance": {"contains_appearance_info", "appearance_info"},
    "subjective_language": {"contains_subjective_language", "subjective_language_info"},
}

terms_files = {
    "physical_health": "./evaluate_themes/txt/physical_health.txt",
    "physical_appearance": "./evaluate_themes/txt/physical_appearance.txt",
    "mental_health": "./evaluate_themes/txt/mental_health.txt",
    "subjective_language": "./evaluate_themes/txt/subjective_language.txt",
}
