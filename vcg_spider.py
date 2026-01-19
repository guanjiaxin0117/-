"""
视觉中国灯具图片爬虫
仅供个人研究学习使用
"""
import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from urllib.parse import urljoin

class VCGImageSpider:
    def __init__(self, output_dir="vcg_lamp_images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置Chrome选项
        self.options = Options()
        self.options.add_argument("--start-maximized")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option('useAutomationExtension', False)
        self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        self.driver = None
        self.downloaded_count = 0
        
    def start_driver(self):
        """启动浏览器"""
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            """
        })
        print("浏览器已启动")
    
    def scroll_page(self, times=3):
        """滚动页面加载更多图片"""
        for i in range(times):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            print(f"页面滚动 {i+1}/{times}")
    
    def get_image_urls(self):
        """获取当前页面的图片URL列表，兼容懒加载和新属性"""
        image_urls = []
        try:
            # 等待图片容器加载（更宽松的选择器）
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "img"))
            )
            # 查找所有img元素
            img_elements = self.driver.find_elements(By.CSS_SELECTOR, "img")
            print(f"调试：本页共发现img标签 {len(img_elements)} 个，打印部分属性：")
            for img in img_elements[:10]:
                print(img.get_attribute('outerHTML'))
            for img in img_elements:
                try:
                    # 依次尝试多种属性
                    src = (
                        img.get_attribute("src") or
                        img.get_attribute("data-src") or
                        img.get_attribute("data-original") or
                        img.get_attribute("data-lazy") or
                        img.get_attribute("data-img")
                    )
                    # 兼容style里background-image
                    if not src:
                        style = img.get_attribute("style")
                        if style and "background-image" in style:
                            import re
                            m = re.search(r'url\(["\']?(.*?)["\']?\)', style)
                            if m:
                                src = m.group(1)
                    if src and src.startswith("//"):
                        src = "https:" + src
                    if src and src.startswith("/") and not src.startswith("//"):
                        src = urljoin(self.driver.current_url, src)
                    if src and "http" in src:
                        if "?" in src:
                            src = src.split("?")[0]
                        if src not in image_urls:
                            image_urls.append(src)
                except Exception as e:
                    continue
            print(f"找到 {len(image_urls)} 个图片URL")
            return image_urls
        except TimeoutException:
            print("页面加载超时")
            return []
    
    def download_image(self, url, filename):
        """下载单张图片"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://www.vcg.com/"
            }
            
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            filepath = os.path.join(self.output_dir, filename)
            
            # 检查文件是否已存在
            if os.path.exists(filepath):
                print(f"文件已存在，跳过: {filename}")
                return True
            
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"✓ 已下载: {filename} ({self.downloaded_count + 1})")
            self.downloaded_count += 1
            return True
            
        except Exception as e:
            print(f"✗ 下载失败 {filename}: {e}")
            return False
    
    def go_to_next_page(self):
        """翻页到下一页"""
        try:
            # 查找下一页按钮（多种选择器）
            next_selectors = [
                "a.next", 
                "a[title='下一页']",
                ".pagination .next",
                ".page-next",
                "li.next a"
            ]
            
            for selector in next_selectors:
                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if next_button and next_button.is_displayed():
                        # 滚动到按钮位置
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                        time.sleep(1)
                        next_button.click()
                        time.sleep(3)
                        print("已翻页到下一页")
                        return True
                except NoSuchElementException:
                    continue
            
            print("未找到下一页按钮")
            return False
            
        except Exception as e:
            print(f"翻页失败: {e}")
            return False
    
    def crawl(self, start_url, max_images=100):
        """主爬取函数"""
        print(f"\n开始爬取: {start_url}")
        print(f"目标数量: {max_images} 张")
        print(f"保存目录: {self.output_dir}\n")
        
        try:
            self.start_driver()
            self.driver.get(start_url)
            
            print("页面加载中，请等待...")
            time.sleep(5)
            
            # 检查是否需要登录
            if "login" in self.driver.current_url.lower():
                print("\n需要登录！")
                print("请在浏览器中手动登录...")
                input("登录完成后，按回车继续...")
                self.driver.get(start_url)
                time.sleep(3)
            
            page_count = 1
            
            while self.downloaded_count < max_images:
                print(f"\n{'='*50}")
                print(f"第 {page_count} 页")
                print(f"{'='*50}")
                
                # 滚动加载图片
                self.scroll_page(3)
                
                # 获取图片URL
                image_urls = self.get_image_urls()
                
                if not image_urls:
                    print("未找到图片，可能需要调整选择器")
                    break
                
                # 下载图片
                for i, url in enumerate(image_urls):
                    if self.downloaded_count >= max_images:
                        break
                    
                    # 生成文件名
                    ext = url.split(".")[-1].split("?")[0]
                    if ext not in ["jpg", "jpeg", "png", "webp"]:
                        ext = "jpg"
                    filename = f"lamp_{self.downloaded_count + 1:04d}.{ext}"
                    
                    self.download_image(url, filename)
                    time.sleep(0.5)  # 防止请求过快
                
                # 检查是否达到目标
                if self.downloaded_count >= max_images:
                    print(f"\n已达到目标数量 {max_images} 张！")
                    break
                
                # 翻页
                if not self.go_to_next_page():
                    print("\n无法继续翻页，爬取结束")
                    break
                
                page_count += 1
                time.sleep(2)
            
            print(f"\n{'='*50}")
            print(f"爬取完成！")
            print(f"成功下载: {self.downloaded_count} 张图片")
            print(f"保存位置: {os.path.abspath(self.output_dir)}")
            print(f"{'='*50}")
            
        except KeyboardInterrupt:
            print("\n用户中断爬取")
        except Exception as e:
            print(f"\n发生错误: {e}")
        finally:
            if self.driver:
                self.driver.quit()
                print("浏览器已关闭")


def main():
    # 配置参数
    TARGET_URL = "https://www.vcg.com/creative-photo/dengju/"
    MAX_IMAGES = 100

    print("="*60)
    print("视觉中国灯具图片爬虫")
    print("仅供个人研究学习使用")
    print("="*60)

    # 输出目录参数已废弃，强制为 vcg_lamp_dataset/未分类
    spider = VCGImageSpider()
    spider.crawl(TARGET_URL, max_images=MAX_IMAGES)
    print("\n图片爬取完成，开始自动训练模型...\n")
    # 自动调用训练脚本
    os.system('python train_after_spider.py')
    print("\n模型训练流程已结束。\n")


if __name__ == "__main__":
    main()
