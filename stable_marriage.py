"""
项目：稳定婚配算法（Stable Marriage Algorithm）
文件：stable_marriage.py（法文版）
描述：使用改进的 Gale–Shapley 算法，法语命名 + 中文注释
"""

import random
import pandas as pd


def generer_preferences(nb_etudiants, nb_etablissements, capacite_par_etablissement=1):
    """
    生成随机偏好
    
    Args:
        nb_etudiants: 学生数量
        nb_etablissements: 院校数量
        capacite_par_etablissement: 每个院校的容量
    
    Returns:
        (学生偏好字典, 院校偏好字典, 容量字典)
    """
    # 生成学生和院校的名称
    etudiants = [f"Etu{i+1}" for i in range(nb_etudiants)]
    etablissements = [f"Eta{j+1}" for j in range(nb_etablissements)]
    
    # 每个院校的容量
    capacites = {etablissement: capacite_par_etablissement for etablissement in etablissements}
    
    # 生成学生对院校的随机偏好
    prefs_etudiants = {e: random.sample(etablissements, len(etablissements)) for e in etudiants}
    
    # 生成院校对学生的随机偏好
    prefs_etablissements = {e: random.sample(etudiants, len(etudiants)) for e in etablissements}
    
    return prefs_etudiants, prefs_etablissements, capacites


def gale_shapley(prefs_etudiants, prefs_etablissements, capacites):
    """
    实现 Gale-Shapley 稳定婚配算法（支持多容量）
    
    算法逻辑：
    - 学生依次向偏好列表中的下一个院校申请；
    - 院校根据自身偏好选择最喜欢的学生；
    - 若院校容量已满，则拒绝最不喜欢的学生；
    - 被拒绝的学生继续申请下一个偏好。
    """
    etudiants_libres = list(prefs_etudiants.keys())  # 初始时所有学生都是自由的
    appariements = {e: [] for e in prefs_etablissements.keys()}  # 存储匹配结果
    propositions_index = {e: 0 for e in prefs_etudiants.keys()}  # 学生申请进度

    while etudiants_libres:
        etu = etudiants_libres.pop(0)
        prefs = prefs_etudiants[etu]
        idx = propositions_index[etu]

        if idx < len(prefs):
            eta = prefs[idx]  # 获取下一个偏好的院校
            propositions_index[etu] = idx + 1
            capacite = capacites[eta]

            # 若院校未满，直接录取
            if len(appariements[eta]) < capacite:
                appariements[eta].append(etu)
            else:
                # 院校已满，比较偏好
                appariements[eta].append(etu)
                appariements[eta].sort(key=lambda x: prefs_etablissements[eta].index(x))
                pire = appariements[eta].pop()  # 拒绝最不喜欢的学生

                if pire != etu:
                    etudiants_libres.append(pire)
                else:
                    etudiants_libres.append(etu)
    return appariements


def calculer_satisfaction(appariements, prefs_etudiants, prefs_etablissements):
    """
    计算满意度
    - 学生满意度 = (总数 - 偏好排名) / 总数
    - 院校满意度同理
    """
    satisfaction_etudiants = []
    satisfaction_etablissements = []

    # 学生满意度
    for etu in prefs_etudiants:
        for eta, etus in appariements.items():
            if etu in etus:
                rang = prefs_etudiants[etu].index(eta)
                satisfaction = (len(prefs_etudiants[etu]) - rang) / len(prefs_etudiants[etu])
                satisfaction_etudiants.append(satisfaction)
                break

    # 院校满意度
    for eta in prefs_etablissements:
        for etu in appariements[eta]:
            rang = prefs_etablissements[eta].index(etu)
            satisfaction = (len(prefs_etablissements[eta]) - rang) / len(prefs_etablissements[eta])
            satisfaction_etablissements.append(satisfaction)

    moy_etu = sum(satisfaction_etudiants) / len(satisfaction_etudiants) if satisfaction_etudiants else 0
    moy_eta = sum(satisfaction_etablissements) / len(satisfaction_etablissements) if satisfaction_etablissements else 0

    return moy_etu, moy_eta


def est_stable(appariements, prefs_etudiants, prefs_etablissements):
    """
    检查匹配是否稳定（是否存在阻塞对）
    
    若存在学生与院校互相更偏好对方而当前未配对，则为不稳定。
    """
    for etu, prefs in prefs_etudiants.items():
        for eta in prefs:
            # 找出学生当前分配的院校
            affecte = None
            for eta_temp, etus_list in appariements.items():
                if etu in etus_list:
                    affecte = eta_temp
                    break

            # 若学生更喜欢另一个院校
            if affecte and prefs.index(eta) < prefs.index(affecte):
                capacite = len(appariements[eta])
                if capacite > 0:
                    pire_etu = appariements[eta][0]
                    pire_rang = prefs_etablissements[eta].index(pire_etu)

                    for s in appariements[eta][1:]:
                        s_rang = prefs_etablissements[eta].index(s)
                        if s_rang > pire_rang:
                            pire_rang = s_rang
                            pire_etu = s

                    # 若院校更喜欢新学生
                    etu_rang = prefs_etablissements[eta].index(etu)
                    if etu_rang < pire_rang:
                        return False, (etu, eta)
    return True, None


def executer_experiences(configs):
    """
    批量执行实验，生成匹配统计结果
    """
    resultats = []

    for cfg in configs:
        nb_etudiants = cfg["nb_etudiants"]
        nb_etablissements = cfg["nb_etablissements"]
        capacite = cfg["capacite"]

        # 生成随机偏好
        pe, petab, caps = generer_preferences(nb_etudiants, nb_etablissements, capacite)

        # 执行 Gale–Shapley 算法
        app = gale_shapley(pe, petab, caps)

        # 计算满意度
        moy_etu, moy_eta = calculer_satisfaction(app, pe, petab)

        # 检查稳定性
        stable, paire = est_stable(app, pe, petab)

        # 保存结果
        resultats.append({
            "id_config": cfg["id"],
            "nb_etudiants": nb_etudiants,
            "nb_etablissements": nb_etablissements,
            "capacite_totale": nb_etablissements * capacite,
            "satisfaction_moy_etudiants": round(moy_etu, 4),
            "satisfaction_moy_etablissements": round(moy_eta, 4),
            "stable": stable,
            "paire_bloquante": paire
        })
    return pd.DataFrame(resultats)
