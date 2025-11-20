import unicodedata
import random


# ------------------------------------------------------------
#  UTILITIES
# ------------------------------------------------------------

def normalize_name(name: str) -> str:
    """Remove accents + unify spacing/casing."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return name.strip().lower()


# ------------------------------------------------------------
#  GALE–SHAPLEY (étudiants → universités)
# ------------------------------------------------------------

def gale_shapley_student_optimal(prefs_students, prefs_universities, capacities):
    """GS étudiant-optimal."""
    free = list(prefs_students.keys())
    next_proposal = {s: 0 for s in prefs_students}
    matching = {u: [] for u in prefs_universities}

    while free:
        stu = free.pop(0)
        prefs = prefs_students[stu]
        if next_proposal[stu] >= len(prefs):
            continue

        uni = prefs[next_proposal[stu]]
        next_proposal[stu] += 1

        if len(matching[uni]) < capacities[uni]:
            matching[uni].append(stu)

        else:
            # Check if the university prefers the new student
            ranking = prefs_universities[uni]
            worst = max(matching[uni], key=lambda x: ranking.index(x))

            if ranking.index(stu) < ranking.index(worst):
                matching[uni].remove(worst)
                matching[uni].append(stu)
                free.append(worst)
            else:
                free.append(stu)
    return matching


# ------------------------------------------------------------
#  GALE–SHAPLEY (universités → étudiants)
# ------------------------------------------------------------

def gale_shapley_university_optimal(prefs_students, prefs_universities, capacities):
    """GS université-optimal."""
    free = list(prefs_universities.keys())
    next_proposal = {u: 0 for u in prefs_universities}
    matching = {u: [] for u in prefs_universities}

    # Invert current matching to student → uni
    student_assignment = {s: None for s in prefs_students}

    while free:
        uni = free.pop(0)
        prefs = prefs_universities[uni]

        if next_proposal[uni] >= len(prefs):
            continue

        stu = prefs[next_proposal[uni]]
        next_proposal[uni] += 1

        current = student_assignment[stu]

        if current is None:
            # Student is free → accept
            student_assignment[stu] = uni
            matching[uni].append(stu)

        else:
            # Student compares universities based on its preference list
            ranking = prefs_students[stu]

            if ranking.index(uni) < ranking.index(current):
                # Student prefers new uni
                matching[current].remove(stu)
                matching[uni].append(stu)
                student_assignment[stu] = uni

                if len(matching[current]) < capacities[current]:
                    free.append(current)

            else:
                free.append(uni)

    return matching


# ------------------------------------------------------------
#  STABILITÉ
# ------------------------------------------------------------

def is_stable(matching, prefs_students, prefs_universities):
    """Check classical stability."""
    # Build reverse map student → uni
    assigned = {}
    for uni, stus in matching.items():
        for s in stus:
            assigned[s] = uni

    for stu, prefs in prefs_students.items():
        stu_assigned = assigned.get(stu, None)

        for uni in prefs:
            if uni == stu_assigned:
                break

            uni_list = matching[uni]
            ranking = prefs_universities[uni]

            # Find worst assigned student at uni
            if uni_list:
                worst = max(uni_list, key=lambda x: ranking.index(x))
                if ranking.index(stu) < ranking.index(worst):
                    return False, (stu, uni)
    return True, None


# ------------------------------------------------------------
#  SATISFACTION
# ------------------------------------------------------------

def satisfaction(matching, prefs_students, prefs_universities):
    sat_students = []
    sat_unis = []

    # Students
    for stu, prefs in prefs_students.items():
        for uni, stus in matching.items():
            if stu in stus:
                rank = prefs.index(uni)
                sat_students.append((len(prefs) - rank) / len(prefs))
                break

    # Universities
    for uni, prefs in prefs_universities.items():
        for stu in matching[uni]:
            rank = prefs.index(stu)
            sat_unis.append((len(prefs) - rank) / len(prefs))

    return sum(sat_students)/len(sat_students), sum(sat_unis)/len(sat_unis)
