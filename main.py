import re

string = "[CLS]7..邓1小2平2.............邓1小2平2....马1克2思2..................................[SEP]7"
# 使用正则表达式匹配数字前的字符
pattern = r'([^0-9]*)([1256])'
result = re.findall(pattern, string)

# 处理匹配结果
output = ""
for group in result:
    output += group[0]
output = re.sub(r'\.+', ' ', output)
print(output)


