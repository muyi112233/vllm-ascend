import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Repo\vllm-ascend\docs\source\tutorials\features\glm-5.1-ascend-support.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix: Remove excessive whitespace in list items
content = re.sub(r'(\n- )\s{10,}', r'\1', content)

# Fix: Remove leading whitespace in list items from docx formatting
content = re.sub(r'^(\s{50,})(- )', r'\2', content, flags=re.MULTILINE)

# Fix: Add code block for tool_call python code
old = '### 调用tool_call时选择auto模型，不要选择required模式\n\nresponse_stream = client.chat.completions.create('
new = '### 调用tool_call时选择auto模型，不要选择required模式\n\n```python\nresponse_stream = client.chat.completions.create('
content = content.replace(old, new)

old = 'tool_choice="auto"       #调用tool时把tool_choice 设置为auto模式，不选required模式\n## A2双机背靠背'
new = 'tool_choice="auto"       #调用tool时把tool_choice 设置为auto模式，不选required模式\n```\n\n## A2双机背靠背'
content = content.replace(old, new)

# Fix: Add code block for chat_template_kwargs and curl
old = '通过如下配置，可以控制输出是否带think：\n"chat_template_kwargs"'
new = '通过如下配置，可以控制输出是否带think：\n\n```json\n"chat_template_kwargs"'
content = content.replace(old, new)

old = '}\n完整请求体参考：\ncurl'
new = '}\n```\n\n完整请求体参考：\n\n```bash\ncurl'
content = content.replace(old, new)

old = "}'\nkwargs配置"
new = "}'\n```\n\nkwargs配置"
content = content.replace(old, new)

# Fix: Add code block for PowerShell script in SHA256 section
old = '修改完成后，将下面这段标红的文字复制到windows powershell，即可获取模型文件的sha256值，并保存在sha256_all_files.txt文档中。\n# 定义路径'
new = '修改完成后，将下面这段标红的文字复制到windows powershell，即可获取模型文件的sha256值，并保存在sha256_all_files.txt文档中。\n\n```powershell\n# 定义路径'
content = content.replace(old, new)

# Fix: Close PowerShell code block
old = 'Write-Host "`n执行完成！仅纯SHA256值已保存到：`n$outputFile"\n\n把PC端'
new = 'Write-Host "`n执行完成！仅纯SHA256值已保存到：`n$outputFile"\n```\n\n把PC端'
content = content.replace(old, new)

# Fix: Add code block for vim command
old = '规避方法：编辑文件\nvim /vllm-workspace'
new = '规避方法：编辑文件\n\n```bash\nvim /vllm-workspace'
content = content.replace(old, new)

old = 'vim /vllm-workspace/vllm-ascend/vllm_ascend/ascend_forward_context.py\n在select_moe_comm_method'
new = 'vim /vllm-workspace/vllm-ascend/vllm_ascend/ascend_forward_context.py\n```\n\n在select_moe_comm_method'
content = content.replace(old, new)

# Fix: Add code block for sha256sum command
old = '在服务器模型文件目录下执行：sha256sum  *'
new = '在服务器模型文件目录下执行：\n\n```bash\nsha256sum  *\n```'
content = content.replace(old, new)

# Fix: Add code block for podman commands in FAQ
old = 'A3镜像下载：Openeuler镜像文件（arm）：\nsudo podman pull'
new = 'A3镜像下载：Openeuler镜像文件（arm）：\n\n```bash\nsudo podman pull'
content = content.replace(old, new)

# Close podman code block after last pull command
old = 'sudo podman pull --platform linux/arm64  quay.io/ascend/vllm-ascend:v0.13.0rc2\n\n'
new = 'sudo podman pull --platform linux/arm64  quay.io/ascend/vllm-ascend:v0.13.0rc2\n```\n\n'
content = content.replace(old, new)

# Fix: Add code block for docker_image_puller
old = 'CMD切换到D盘后直接执行\n.\\\\docker_image_puller.exe'
new = 'CMD切换到D盘后直接执行\n\n```bash\n.\\\\docker_image_puller.exe'
content = content.replace(old, new)

old = '.\\\\docker_image_puller.exe -i quay.io/ascend/vllm-ascend:v0.13.0rc2 -a arm64下载的就是arm架构的镜像文件'
new = '.\\\\docker_image_puller.exe -i quay.io/ascend/vllm-ascend:v0.13.0rc2 -a arm64\n```\n\n下载的就是arm架构的镜像文件'
content = content.replace(old, new)

# Fix: Add code block for Enable-WindowsOptionalFeature
old = '- 开启虚拟化配置\nEnable-WindowsOptionalFeature'
new = '- 开启虚拟化配置\n\n```powershell\nEnable-WindowsOptionalFeature'
content = content.replace(old, new)

old = 'Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All\n执行完会重启'
new = 'Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All\n```\n\n执行完会重启'
content = content.replace(old, new)

# Fix: Add code block for wsl install
old = '- 虚拟化"已启用"后，需要安装wsl\n在管理员权限下登入windows powershell，执行wsl --install'
new = '- 虚拟化"已启用"后，需要安装wsl\n\n在管理员权限下登入windows powershell，执行：\n\n```powershell\nwsl --install\n```'
content = content.replace(old, new)

# Fix: Add code block for sudo apt commands
old = '- 在虚拟系统执行sudo apt update && sudo apt upgrade -y来升级系统软件'
new = '- 在虚拟系统执行以下命令来升级系统软件：\n\n```bash\nsudo apt update && sudo apt upgrade -y\n```'
content = content.replace(old, new)

old = '- 升级完系统软件后，执行sudo apt install podman -y来安装podman'
new = '- 升级完系统软件后，执行以下命令来安装podman：\n\n```bash\nsudo apt install podman -y\n```'
content = content.replace(old, new)

# Fix: Add code block for podman save
old = '- 将下载好的镜像保存出来（从虚拟系统保存到pc的桌面）\nsudo podman save'
new = '- 将下载好的镜像保存出来（从虚拟系统保存到pc的桌面）\n\n```bash\nsudo podman save'
content = content.replace(old, new)

old = 'quay.io/ascend/vllm-ascend:v0.14.0rc1-a3-openeuler-----podman系统中镜像的repository:tag'
new = 'quay.io/ascend/vllm-ascend:v0.14.0rc1-a3-openeuler-----podman系统中镜像的repository:tag\n```'
content = content.replace(old, new)

# Fix: Add code block for podman images
old = '- 查看镜像文件：sudo podman images'
new = '- 查看镜像文件：\n\n```bash\nsudo podman images\n```'
content = content.replace(old, new)

# Fix: Add code block for pip list
old = '（1）查询aisbench的安装目录\npip list |grep ais'
new = '（1）查询aisbench的安装目录\n\n```bash\npip list |grep ais\n```'
content = content.replace(old, new)

# Fix: Add code block for ais_bench search
old = 'ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf –search'
new = '```bash\nais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --search\n```'
content = content.replace(old, new)

# Fix: Add code block for ais_bench test
old = 'ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_0_shot_cot_chat_prompt --merge-ds –debug'
new = '```bash\nais_bench --models vllm_api_general_chat --datasets gsm8k_gen_0_shot_cot_chat_prompt --merge-ds --debug\n```'
content = content.replace(old, new)

# Fix: Add code block for vim in accuracy test
old = 'vim  aisbench安装路径/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py'
new = '```bash\nvim  aisbench安装路径/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py\n```'
content = content.replace(old, new)

# Fix: Add code block for wget in accuracy test
old = 'cd ais_bench/datasets\nwget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip\nunzip gsm8k.zip\nrm gsm8k.zip'
new = '```bash\ncd ais_bench/datasets\nwget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip\nunzip gsm8k.zip\nrm gsm8k.zip\n```'
content = content.replace(old, new)

# Fix: Add code block for hccn_tool commands in A3 section
old = '1，查看NPU状态，在每个服务器节点执行：\nnpu-smi info'
new = '1，查看NPU状态，在每个服务器节点执行：\n\n```bash\nnpu-smi info\n```'
content = content.replace(old, new)

# Fix: Remove excessive whitespace lines
content = re.sub(r'\n{4,}', '\n\n\n', content)

# Fix: Remove excessive whitespace in "检查HDK版本" section
old = 'HDK建议采用25.5.0版本\n版本配套信息如下。转测版本镜像除HDK外，其他配套组件均已打包至镜像中，仅需在宿主机裸机环境安装配套HDK。\n                                                                                                                    - 检查HDK版本'
new = 'HDK建议采用25.5.0版本\n版本配套信息如下。转测版本镜像除HDK外，其他配套组件均已打包至镜像中，仅需在宿主机裸机环境安装配套HDK。\n\n- 检查HDK版本'
content = content.replace(old, new)

with open(r'C:\Repo\vllm-ascend\docs\source\tutorials\features\glm-5.1-ascend-support.md', 'w', encoding='utf-8') as f:
    f.write(content)

print('Cleanup done')
