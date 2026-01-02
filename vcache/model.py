
import re
import time
from openai import OpenAI, APIError  # 导入 time 和 openai 的 APIError

class SiliconFlowLLM:
    """ 
    使用 OpenAI SDK 连接阿里云百炼平台，并集成了指数退避重试机制。
    (类名保持不变以符合您的要求)
    """
    def __init__(self, api_key, model="deepseek-r1-distill-qwen-14b"):
        # 这部分完全不变
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        self.category_regex = re.compile(
            r'(?i)\b(Book[ s]?|Electronics|Household|Clothing\s*[&]?\s*Accessories)',
            re.IGNORECASE
        )

    def generate_response(self, prompt):
        """ 
        执行分类流程，并内置了针对服务器繁忙的自动重试逻辑。
        """
        max_attempts = 4  # 最大尝试次数 (1次正常 + 3次重试)
        backoff_factor = 2  # 等待时间的基数（秒）

        for attempt in range(max_attempts):
            try:
                # 添加API调用日志（保持不变）
                print(f"[API Request] Prompt: {prompt[:100]}...") 
                
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    top_p=0.9,
                    max_tokens=1024,
                    n=1
                )
                raw_response = completion.choices[0].message.content
                
                # 记录原始响应（保持不变）
                print(f"[API Response] Raw: {raw_response}")  
                
                # 如果成功，清理响应并直接返回结果，跳出循环
                return self._clean_response(raw_response)

            except APIError as e:
                # 只捕获 openai 库的 API 错误，这些通常是网络或服务器端问题
                # 检查错误信息是否是可重试的类型
                error_message = str(e).lower()
                if ('serviceunavailable' in error_message or 
                    'throttled' in error_message or 
                    'too many requests' in error_message or
                    'internalerror' in error_message):
                    
                    # 如果是最后一次尝试，则不再等待，直接抛出错误
                    if attempt == max_attempts - 1:
                        print(f"API请求失败，已达到最大重试次数 ({max_attempts}次)。")
                        raise RuntimeError(f"API处理失败: {str(e)}") from e
                    
                    # 计算下一次重试的等待时间（指数退避）
                    wait_time = backoff_factor * (2 ** attempt)
                    print(f"服务器繁忙，API请求失败。将在 {wait_time} 秒后进行第 {attempt + 1} 次重试...")
                    time.sleep(wait_time)

                else:
                    # 如果是其他API错误（如认证失败401），则不重试，直接抛出
                    raise RuntimeError(f"API处理失败，发生不可重试错误: {str(e)}") from e
            
            except Exception as e:
                # 捕获其他所有意料之外的错误，直接抛出
                raise RuntimeError(f"发生未知错误: {str(e)}") from e

    def _clean_response(self, text):
        """严格匹配四类商品，自动修正单复数（保持不变）"""
        cleaned_text = text.strip()
        if not cleaned_text:
            return "Unknown"
        
        pattern = r'''
        (?xi)
        (?:.*?\b(?:Category\s*:\s*)?){0,1}
        \b(Books?|Electronics|Household|Clothing\s*[&+]?\s*Accessories)\b
        (?:[.,;!]?|\n|$)
        '''
        match = re.search(pattern, cleaned_text, re.VERBOSE)
        if match:
            raw = match.group(1).lower().replace(' ', '')
            return self._format_category(raw)
        return "Unknown"

    def _format_category(self, raw):
        """强制输出标准四类（保持不变），使用.get()更安全"""
        format_map = {
            'book': 'Books',
            'books': 'Books',
            'electronics': 'Electronics',
            'household': 'Household',
            'clothing&accessories': 'Clothing & Accessories',
            'clothing+accessories': 'Clothing & Accessories',
            'clothingandaccessories': 'Clothing & Accessories',
            'clothingaccessories': 'Clothing & Accessories'    
        }
        # 使用 .get() 避免因模型返回意外内容导致 KeyError
        return format_map.get(raw, "Unknown")

# ✅ 测试用例（保持不变）
if __name__ == "__main__":
    test_prompt = """请分类以下商品：
商品描述：Wowobjects乒乓球橡胶配件
可选类别：Books, Electronics, Household, Clothing & Accessories
要求：只输出类别名称，不要任何符号"""

    # 请在这里替换为你的真实阿里云百炼 API Key
    api_key = "sk-your-real-api-key"

    if api_key == "sk-your-real-api-key":
        print("错误：请在代码中替换 'sk-your-real-api-key' 为你自己的 API Key。")
    else:
        try:
            llm = SiliconFlowLLM(api_key=api_key)
            print("正在调用API进行分类...")
            
            start_time = time.time()
            category = llm.generate_response(test_prompt)
            elapsed_time = time.time() - start_time

            print("\n--------------------")
            print(f"最终分类结果: {category}")
            print(f"总耗时: {elapsed_time:.2f}秒")
            print("--------------------")

        except Exception as e:
            print(f"\n最终错误捕获: {str(e)}")