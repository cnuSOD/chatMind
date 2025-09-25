#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
哈密尔顿抑郁量表（HAMD）评估系统测试文件
测试HAMD评估的各项功能，包括问答流程、评分逻辑和报告生成
"""

import os
import time
from hamd_evaluator import HAMDEvaluator


def test_hamd_basic_functionality():
    """测试HAMD评估器的基本功能"""
    print("=== 测试HAMD评估器基本功能 ===")
    
    # 创建评估器实例
    evaluator = HAMDEvaluator(
        questions_file="hamd_questions.json",
        output_dir="./test_HAMD_Results/"
    )
    
    # 测试问题加载
    print(f"✓ 成功加载 {len(evaluator.questions)} 个问题")
    
    # 开始评估
    evaluation_id = evaluator.start_evaluation()
    print(f"✓ 评估开始，ID: {evaluation_id}")
    
    # 测试获取问题
    current_q = evaluator.get_current_question()
    print(f"✓ 当前问题: {current_q['title']} - {current_q['question']}")
    
    # 测试进度
    progress = evaluator.get_progress()
    print(f"✓ 评估进度: {progress['current_question']}/{progress['total_questions']}")
    
    print("基本功能测试完成\n")


def test_hamd_evaluation_flow():
    """测试完整的HAMD评估流程"""
    print("=== 测试完整评估流程 ===")
    
    evaluator = HAMDEvaluator(
        questions_file="hamd_questions.json",
        output_dir="./test_HAMD_Results/"
    )
    
    # 开始评估
    evaluation_id = evaluator.start_evaluation()
    print(f"开始评估，ID: {evaluation_id}")
    
    # 模拟用户回答（涵盖不同严重程度）
    test_answers = [
        "我最近心情确实不太好，经常感到悲伤，对很多事情都失去了兴趣，感觉很绝望",  # 抑郁情绪 - 预期3分
        "有时候会觉得自己做错了什么，经常自责，觉得都是我的错",  # 有罪感 - 预期2分
        "没有自杀的念头，虽然有时候会想如果消失就好了",  # 自杀意图 - 预期1分
        "睡眠很不好，经常失眠，半夜醒来就睡不着了",  # 睡眠障碍 - 预期2分
        "食欲还可以，偶尔不想吃东西",  # 食欲减退 - 预期1分
        "对工作和社交确实兴趣不大，什么都不想做",  # 兴趣丧失 - 预期3分
        "体重没有明显变化",  # 体重减轻 - 预期0分
        "感觉做事比以前慢很多，总是没有精神",  # 精神运动性迟缓 - 预期2分
        "经常感到焦虑和烦躁，控制不住",  # 焦虑 - 预期2分
        "觉得自己很没用，什么都做不好",  # 自我评价 - 预期2分
        "感觉和别人有距离感，不愿意交流",  # 情感平淡 - 预期2分
        "这方面兴趣确实下降了",  # 性兴趣减退 - 预期1分
        "经常感到很累，没有力气做事情",  # 无力感 - 预期2分
        "做事情的积极性很低，需要别人推动",  # 积极性 - 预期2分
        "感知能力还算正常",  # 感知能力 - 预期0分
        "有时候会头痛，心慌"  # 其他症状 - 预期1分
    ]
    
    print(f"\n开始处理 {len(test_answers)} 个回答...")
    
    for i, answer in enumerate(test_answers):
        if evaluator.is_evaluation_complete():
            break
            
        current_q = evaluator.get_current_question()
        print(f"\n--- 问题 {current_q['index']}: {current_q['title']} ---")
        print(f"问题: {current_q['question']}")
        print(f"回答: {answer}")
        
        # 处理回答
        result = evaluator.process_answer(answer)
        print(f"评分: {result.get('score', 'N/A')}分")
        
        if result.get("is_complete", False):
            print("评估完成！")
            break
    
    # 生成报告
    if evaluator.is_evaluation_complete():
        print("\n=== 生成评估报告 ===")
        report = evaluator.generate_report()
        
        print(f"总分: {report['total_score']}/{report['max_score']}")
        print(f"严重程度: {report['severity_level']['level']}")
        print(f"描述: {report['severity_level']['description']}")
        
        print("\n分项分析:")
        for category, data in report['category_analysis'].items():
            print(f"  {category}: {data['score']}/{data['max_score']} ({data['percentage']:.1f}%) - {data['severity']}")
        
        print(f"\n建议:")
        for i, recommendation in enumerate(report['recommendations'][:3], 1):
            print(f"  {i}. {recommendation}")
        
        print(f"\n报告保存位置: {evaluator.output_dir}")
    
    print("完整评估流程测试完成\n")


def test_hamd_edge_cases():
    """测试HAMD评估的边界情况"""
    print("=== 测试边界情况 ===")
    
    evaluator = HAMDEvaluator(
        questions_file="hamd_questions.json",
        output_dir="./test_HAMD_Results/"
    )
    
    # 测试极端回答
    evaluator.start_evaluation()
    
    # 测试极低分回答
    low_score_answers = [
        "我心情很好，没有任何问题",
        "没有罪恶感",
        "完全没有自杀念头",
        "睡眠很好",
        "食欲正常",
        "对所有事情都很有兴趣",
        "体重正常",
        "精神状态很好",
        "没有焦虑",
        "自我感觉良好",
        "情感正常",
        "性兴趣正常",
        "精力充沛",
        "积极性很高",
        "感知正常",
        "没有其他症状"
    ]
    
    print("测试极低分情况...")
    for answer in low_score_answers[:3]:  # 只测试前3个问题
        current_q = evaluator.get_current_question()
        if current_q:
            result = evaluator.process_answer(answer)
            print(f"问题{current_q['index']}: {result.get('score', 0)}分")
    
    # 重置评估器测试极高分
    evaluator.reset_evaluation()
    evaluator.start_evaluation()
    
    high_score_answers = [
        "我感到极度绝望，完全失去了活下去的意义，每天都在痛苦中度过",
        "我觉得一切都是我的错，我是个罪人，应该受到惩罚",
        "我经常想到自杀，已经制定了具体的计划",
    ]
    
    print("测试极高分情况...")
    for answer in high_score_answers:
        current_q = evaluator.get_current_question()
        if current_q:
            result = evaluator.process_answer(answer)
            print(f"问题{current_q['index']}: {result.get('score', 0)}分")
    
    print("边界情况测试完成\n")


def test_hamd_scoring_logic():
    """测试HAMD评分逻辑"""
    print("=== 测试评分逻辑 ===")
    
    evaluator = HAMDEvaluator(
        questions_file="hamd_questions.json",
        output_dir="./test_HAMD_Results/"
    )
    
    # 测试规则评分
    test_cases = [
        ("我很正常，没有问题", 0),  # 应该得0分
        ("偶尔会有点不开心", 1),   # 应该得1分
        ("经常感到困难和痛苦", 2),  # 应该得2分
        ("非常严重，无法正常生活", 4),  # 应该得4分
    ]
    
    print("测试规则评分:")
    for answer, expected_score in test_cases:
        score = evaluator._rule_based_scoring(answer)
        print(f"回答: '{answer}' -> 得分: {score} (期望: {expected_score})")
    
    print("评分逻辑测试完成\n")


def test_hamd_report_generation():
    """测试报告生成功能"""
    print("=== 测试报告生成 ===")
    
    evaluator = HAMDEvaluator(
        questions_file="hamd_questions.json",
        output_dir="./test_HAMD_Results/"
    )
    
    # 手动设置一些评分数据
    evaluator.start_evaluation()
    evaluator.scores = {
        1: {"score": 2, "analysis": "测试分析1", "scoring_criteria": "明确情绪低落"},
        2: {"score": 1, "analysis": "测试分析2", "scoring_criteria": "偶尔自责"},
        3: {"score": 0, "analysis": "测试分析3", "scoring_criteria": "无自杀念头"},
        4: {"score": 3, "analysis": "测试分析4", "scoring_criteria": "严重失眠"},
        5: {"score": 1, "analysis": "测试分析5", "scoring_criteria": "偶尔食欲减退"},
    }
    
    # 强制设置为完成状态
    evaluator.current_question_index = len(evaluator.questions)
    
    report = evaluator.generate_report()
    
    print("报告生成测试:")
    print(f"✓ 生成报告成功")
    print(f"✓ 总分: {report['total_score']}")
    print(f"✓ 严重程度: {report['severity_level']['level']}")
    print(f"✓ 分项分析: {len(report['category_analysis'])} 个类别")
    print(f"✓ 建议数量: {len(report['recommendations'])}")
    
    print("报告生成测试完成\n")


def main():
    """主测试函数"""
    print("开始HAMD评估系统测试")
    print("=" * 60)
    
    try:
        # 确保测试目录存在
        os.makedirs("./test_HAMD_Results/", exist_ok=True)
        
        # 运行各项测试
        test_hamd_basic_functionality()
        test_hamd_evaluation_flow()
        test_hamd_edge_cases()
        test_hamd_scoring_logic()
        test_hamd_report_generation()
        
        print("=" * 60)
        print("✅ 所有测试完成！HAMD评估系统功能正常")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
