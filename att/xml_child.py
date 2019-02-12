import re


def get_child(all_children):
    all_third_lvl_children = []
    # print(type(all_children))
    # print(all_children)
    for each_child in all_children:
        all_third_lvl_children.append(each_child)

        if len(each_child['children']) > 0:
            all_third_lvl_children.extend(get_child(each_child['children']))
            each_child['children'] = []
    return all_third_lvl_children


def create_3_lvl_relation(all_fields):
    child_list = []
    for each_field in all_fields:
        for each_child in each_field['children']:
            if len(each_child) > 0:
                child_list.append(get_child(each_child['children']))
    return child_list


res = {
    "all_Fields": [
        {
            "id": "1",
            "tag": "InvoiceNumber",
            "type": "Key-value pair",
            "key": "invoice number",
            "value": "361685",
            "children": [
                {
                    "id": "2",
                    "tag": "total_current_charges",
                    "type": "Key-value pair",
                    "key": "invoice number",
                    "value": "361685",
                    "children": [
                        {
                            "id": "3",
                            "tag": "total_current_charges",
                            "type": "Key-value pair",
                            "key": "invoice number",
                            "value": "361685",
                            "children": [
                                {
                                    "id": "4",
                                    "tag": "total_current_charges",
                                    "type": "Key-value pair",
                                    "key": "invoice number",
                                    "value": "361685",
                                    "children": [
                                        {
                                            "id": "5",
                                            "tag": "total_current_charges",
                                            "type": "Key-value pair",
                                            "key": "invoice number",
                                            "value": "361685",
                                            "children": []
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                },
                {
                    "id": "3",
                    "tag": "AccountNumber",
                    "type": "Key-value pair",
                    "key": "account number",
                    "value": "112227",
                    "children": [
                        {
                            "id": "33",
                            "tag": "AccountNumber",
                            "type": "Key-value pair",
                            "key": "account number",
                            "value": "112227",
                            "children": []
                        }
                    ]
                }
            ]
        }
    ],
    "page_Data": [
        {
            "pageNumber": 0,
            "confidenceScore": 0
        },
        {
            "pageNumber": -1,
            "confidenceScore": 1
        }
    ]
}
lvl3_res = create_3_lvl_relation(res['all_Fields'])
print(lvl3_res)

invoice_info_list = ['total_current_charges', 'late_charges', 'PDB']
fields = res["all_Fields"]
header = '<invoice '
invoice_infos = []
invoice_details = []
for f in fields:
    header += f['tag'] + '="' + f['value'] + '" '
    for f1 in f['children']:
        child_tag = '<charge '
        if f1['tag'] in invoice_info_list:
            child_tag += 'amount="' + f1['value'] + '" type="' + f1['tag'] + '"/>'
            invoice_infos.append(child_tag)
        else:
            child_tag += 'amount="' + f1['value'] + '" description="' + f1['tag'] + \
                         '" eng_description="' + f1['tag'] + '"/>'
            invoice_details.append(child_tag)
header = re.sub(' $', '>', header)

for l in lvl3_res:
    for entry in l:
        child_tag = '<charge '
        child_tag += 'amount="' + entry['value'] + '" description="' + entry['tag'] + \
                     '" eng_description="' + entry['tag'] + '"/>'
        invoice_details.append(child_tag)

xml_output = header
if invoice_infos:
    xml_output += '<invoice_info>'
    for info in invoice_infos:
        xml_output += info
        xml_output += '</invoice_info>'
if invoice_details:
    xml_output += '<invoice_details>'
    xml_output += '<line item="">'
    for detail in invoice_details:
        xml_output += detail
    xml_output += '</line>'
    xml_output += '</invoice_details>'
xml_output += '</invoice>'
print(xml_output)
with open('/home/rztuser/test.xml', "w") as fs:
    fs.write(xml_output)
