"""
交互界面模块
提供命令行和简单的Web界面
"""

import argparse
import sys
import os
from inference import LLMInference
from trainer import LLMTrainer
from data_preprocessor import DataPreprocessor
from config import Config


class CommandLineInterface:
    """命令行界面"""
    
    def __init__(self):
        self.inference = None
        self.trainer = None
    
    def parse_arguments(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(description='LLM演示程序')
        
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # 训练命令
        train_parser = subparsers.add_parser('train', help='训练模型')
        train_parser.add_argument('--data', type=str, default=Config.DATA_PATH, 
                                help='训练数据文件路径')
        train_parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                                help='训练轮数')
        train_parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                                help='批次大小')
        train_parser.add_argument('--save_path', type=str, default=Config.MODEL_SAVE_PATH,
                                help='模型保存路径')
        
        # 推理命令
        infer_parser = subparsers.add_parser('infer', help='使用模型推理')
        infer_parser.add_argument('--model', type=str, default=Config.MODEL_SAVE_PATH,
                                help='模型文件路径')
        infer_parser.add_argument('--vocab', type=str, default=Config.VOCAB_SAVE_PATH,
                                help='词汇表文件路径')
        infer_parser.add_argument('--prompt', type=str, required=True,
                                help='输入提示文本')
        infer_parser.add_argument('--max_length', type=int, default=50,
                                help='最大生成长度')
        infer_parser.add_argument('--temperature', type=float, default=0.8,
                                help='温度参数')
        infer_parser.add_argument('--top_k', type=int, default=50,
                                help='Top-k采样')
        
        # 交互命令
        interactive_parser = subparsers.add_parser('interactive', help='交互式生成')
        interactive_parser.add_argument('--model', type=str, default=Config.MODEL_SAVE_PATH,
                                       help='模型文件路径')
        interactive_parser.add_argument('--vocab', type=str, default=Config.VOCAB_SAVE_PATH,
                                       help='词汇表文件路径')
        
        # 评估命令
        eval_parser = subparsers.add_parser('evaluate', help='评估模型')
        eval_parser.add_argument('--model', type=str, default=Config.MODEL_SAVE_PATH,
                                help='模型文件路径')
        eval_parser.add_argument('--vocab', type=str, default=Config.VOCAB_SAVE_PATH,
                                help='词汇表文件路径')
        eval_parser.add_argument('--data', type=str, default=Config.DATA_PATH,
                                help='评估数据文件路径')
        
        return parser.parse_args()
    
    def load_data(self, filepath):
        """加载数据文件"""
        if not os.path.exists(filepath):
            print(f"数据文件不存在: {filepath}")
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"加载了 {len(texts)} 条文本数据")
        return texts
    
    def train_command(self, args):
        """训练命令处理"""
        print("=== LLM模型训练 ===")
        
        # 加载数据
        texts = self.load_data(args.data)
        if not texts:
            print("没有可用的训练数据")
            return
        
        # 初始化训练器
        self.trainer = LLMTrainer()
        self.trainer.num_epochs = args.epochs
        self.trainer.batch_size = args.batch_size
        
        # 开始训练
        self.trainer.train(texts, args.save_path)
        
        # 显示训练历史
        self.trainer.plot_training_history()
    
    def infer_command(self, args):
        """推理命令处理"""
        print("=== LLM文本生成 ===")
        
        # 初始化推理器
        self.inference = LLMInference(args.model, args.vocab)
        
        # 生成文本
        results = self.inference.generate_text(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        print("\n生成完成!")
        return results
    
    def interactive_command(self, args):
        """交互命令处理"""
        print("=== LLM交互式文本生成 ===")
        
        # 初始化推理器
        self.inference = LLMInference(args.model, args.vocab)
        
        # 显示模型信息
        model_info = self.inference.get_model_info()
        print(f"模型信息: {model_info}")
        
        # 开始交互
        self.inference.interactive_generation()
    
    def evaluate_command(self, args):
        """评估命令处理"""
        print("=== LLM模型评估 ===")
        
        # 加载数据
        texts = self.load_data(args.data)
        if not texts:
            print("没有可用的评估数据")
            return
        
        # 初始化推理器
        self.inference = LLMInference(args.model, args.vocab)
        
        # 初始化训练器用于评估
        self.trainer = LLMTrainer(self.inference.model, self.inference.data_preprocessor)
        
        # 评估模型
        eval_results = self.trainer.evaluate_model(texts)
        
        print("\n评估结果:")
        for key, value in eval_results.items():
            print(f"{key}: {value}")
    
    def run(self):
        """运行命令行界面"""
        args = self.parse_arguments()
        
        if not args.command:
            print("请指定一个命令，使用 --help 查看帮助")
            return
        
        try:
            if args.command == 'train':
                self.train_command(args)
            elif args.command == 'infer':
                self.infer_command(args)
            elif args.command == 'interactive':
                self.interactive_command(args)
            elif args.command == 'evaluate':
                self.evaluate_command(args)
            else:
                print(f"未知命令: {args.command}")
                
        except Exception as e:
            print(f"执行命令时出错: {e}")
            import traceback
            traceback.print_exc()


class SimpleWebInterface:
    """简单的Web界面（可选）"""
    
    def __init__(self, model_path=None, vocab_path=None):
        self.inference = LLMInference(model_path, vocab_path)
    
    def generate_html(self):
        """生成简单的HTML界面"""
        html = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM演示程序</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .input-group { margin: 20px 0; }
                textarea, input, button { width: 100%; padding: 10px; margin: 5px 0; }
                .result { background: #f5f5f5; padding: 15px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>LLM文本生成演示</h1>
                
                <div class="input-group">
                    <label for="prompt">输入提示:</label>
                    <textarea id="prompt" rows="3" placeholder="请输入文本提示..."></textarea>
                </div>
                
                <div class="input-group">
                    <label>生成参数:</label>
                    <input type="number" id="max_length" placeholder="最大长度" value="50">
                    <input type="number" id="temperature" step="0.1" placeholder="温度" value="0.8">
                    <input type="number" id="top_k" placeholder="Top-k" value="50">
                </div>
                
                <button onclick="generateText()">生成文本</button>
                
                <div id="result" class="result"></div>
            </div>
            
            <script>
                async function generateText() {
                    const prompt = document.getElementById('prompt').value;
                    const maxLength = document.getElementById('max_length').value;
                    const temperature = document.getElementById('temperature').value;
                    const topK = document.getElementById('top_k').value;
                    
                    if (!prompt) {
                        alert('请输入提示文本');
                        return;
                    }
                    
                    const response = await fetch('/generate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            prompt: prompt,
                            max_length: parseInt(maxLength),
                            temperature: parseFloat(temperature),
                            top_k: parseInt(topK)
                        })
                    });
                    
                    const result = await response.json();
                    document.getElementById('result').innerHTML = 
                        '<strong>生成结果:</strong><br>' + result.generated_text;
                }
            </script>
        </body>
        </html>
        '''
        return html
    
    def start_web_server(self, host='localhost', port=8080):
        """启动简单的Web服务器（需要安装flask）"""
        try:
            from flask import Flask, request, jsonify
            
            app = Flask(__name__)
            
            @app.route('/')
            def index():
                return self.generate_html()
            
            @app.route('/generate', methods=['POST'])
            def generate():
                data = request.json
                results = self.inference.generate_text(
                    prompt=data['prompt'],
                    max_length=data.get('max_length', 50),
                    temperature=data.get('temperature', 0.8),
                    top_k=data.get('top_k', 50)
                )
                
                return jsonify({
                    'generated_text': results[0] if results else ''
                })
            
            print(f"启动Web服务器: http://{host}:{port}")
            app.run(host=host, port=port, debug=False)
            
        except ImportError:
            print("未安装Flask，无法启动Web服务器")
            print("请安装: pip install flask")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 命令行模式
        cli = CommandLineInterface()
        cli.run()
    else:
        # 交互式模式
        print("=== LLM演示程序 ===")
        print("1. 训练模型")
        print("2. 加载模型并生成文本")
        print("3. 交互式生成")
        print("4. 评估模型")
        print("5. 启动Web界面")
        print("6. 退出")
        
        while True:
            try:
                choice = input("\n请选择操作 (1-6): ").strip()
                
                if choice == '1':
                    # 训练模型
                    data_path = input("训练数据文件路径 (默认: data/sample_data.txt): ") or Config.DATA_PATH
                    save_path = input("模型保存路径 (默认: models/llm_demo_model.pth): ") or Config.MODEL_SAVE_PATH
                    
                    cli = CommandLineInterface()
                    cli.train_command(argparse.Namespace(
                        data=data_path,
                        save_path=save_path,
                        epochs=Config.NUM_EPOCHS,
                        batch_size=Config.BATCH_SIZE
                    ))
                    
                elif choice == '2':
                    # 加载模型并生成
                    model_path = input("模型文件路径 (默认: models/llm_demo_model.pth): ") or Config.MODEL_SAVE_PATH
                    vocab_path = input("词汇表文件路径 (默认: models/vocab.pkl): ") or Config.VOCAB_SAVE_PATH
                    prompt = input("请输入提示文本: ")
                    
                    cli = CommandLineInterface()
                    cli.infer_command(argparse.Namespace(
                        model=model_path,
                        vocab=vocab_path,
                        prompt=prompt,
                        max_length=50,
                        temperature=0.8,
                        top_k=50
                    ))
                    
                elif choice == '3':
                    # 交互式生成
                    model_path = input("模型文件路径 (默认: models/llm_demo_model.pth): ") or Config.MODEL_SAVE_PATH
                    vocab_path = input("词汇表文件路径 (默认: models/vocab.pkl): ") or Config.VOCAB_SAVE_PATH
                    
                    cli = CommandLineInterface()
                    cli.interactive_command(argparse.Namespace(
                        model=model_path,
                        vocab=vocab_path
                    ))
                    
                elif choice == '4':
                    # 评估模型
                    model_path = input("模型文件路径 (默认: models/llm_demo_model.pth): ") or Config.MODEL_SAVE_PATH
                    vocab_path = input("词汇表文件路径 (默认: models/vocab.pkl): ") or Config.VOCAB_SAVE_PATH
                    data_path = input("评估数据文件路径 (默认: data/sample_data.txt): ") or Config.DATA_PATH
                    
                    cli = CommandLineInterface()
                    cli.evaluate_command(argparse.Namespace(
                        model=model_path,
                        vocab=vocab_path,
                        data=data_path
                    ))
                    
                elif choice == '5':
                    # Web界面
                    model_path = input("模型文件路径 (默认: models/llm_demo_model.pth): ") or Config.MODEL_SAVE_PATH
                    vocab_path = input("词汇表文件路径 (默认: models/vocab.pkl): ") or Config.VOCAB_SAVE_PATH
                    
                    web_interface = SimpleWebInterface(model_path, vocab_path)
                    web_interface.start_web_server()
                    
                elif choice == '6':
                    print("退出程序")
                    break
                else:
                    print("无效选择，请重新输入")
                    
            except KeyboardInterrupt:
                print("\n退出程序")
                break
            except Exception as e:
                print(f"操作失败: {e}")


if __name__ == "__main__":
    main()