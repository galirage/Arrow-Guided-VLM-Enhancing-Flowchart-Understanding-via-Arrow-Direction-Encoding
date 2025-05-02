common_system_prompt = """
You are a helpful assistant.
"""

original_human_prompt = """
Please answer the following questions using the following flow-chart diagram.
{question_prompt}
"""

dec_ocr_human_prompt = """
Please answer the following questions using the following flow-chart diagram.
{question_prompt}
Refer to the following flow-chart diagram readings by the detection model.
'before object' and 'after object' may be sometimes incorrect and should be used only as a reference.
{flow_chart_text}
"""

question_type_1_prompt = """
In this flowchart diagram, what is the next step after {q_ref_step1}?.
"""

question_type_2_prompt = """
In this flowchart diagram, if {q_ref_step1} is {q_ref_yes_no}, what is the next step?.
"""

question_type_3_prompt = """
In the flowchart diagram, which of the steps before an object {q_ref_step1} except {q_ref_step2}?.
"""

answer_type_1_prompt = """
The next step after {q_ref_step1} is {q_ref_step2}.
"""

answer_type_2_prompt = """
If {q_ref_step1} is {q_ref_yes_no}, the next step is {q_ref_step2}.
"""

answer_type_3_prompt = """
The next step after {q_ref_step1} is {q_ref_step2}.
"""
