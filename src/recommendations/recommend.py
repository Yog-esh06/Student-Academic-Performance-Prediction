"""Rule-based recommendations."""

def generate_recommendations(top_features: list[dict], risk_category: str) -> list[str]:
    """
    Generates 3-5 actionable recommendations based on SHAP feature contributions
    and the student's overall risk category.
    """
    recommendations = set()
    
    # 1. Feature-specific rules (trigger if a feature negatively impacts the score)
    for item in top_features:
        feat = item["feature"]
        contrib = item["shap_contribution"]
        
        # Only trigger advice if the feature is actively dragging the score down
        if contrib < -0.2:
            if "Attendance" in feat:
                recommendations.add("Your attendance rate is significantly impacting your score. Prioritize showing up to all classes.")
            elif "Hours_Studied" in feat:
                recommendations.add("Consider increasing your weekly study hours. A consistent study schedule could quickly boost your performance.")
            elif "Sleep_Hours" in feat:
                recommendations.add("Your sleep schedule is negatively affecting your performance. Aim for a consistent 7-9 hours of rest.")
            elif "Tutoring_Sessions" in feat:
                recommendations.add("You could benefit from extra academic support. Consider attending more tutoring sessions.")
            elif "Parental_Involvement" in feat or "Motivation_Level" in feat:
                recommendations.add("Consider reaching out to a school counselor, mentor, or study group to build a stronger support system.")
            elif "Access_to_Resources" in feat:
                recommendations.add("Try utilizing school resources like the library, online portals, or labs more frequently.")
            elif "Previous_Scores" in feat:
                recommendations.add("Past performance is dragging your projection down. Focus on reviewing foundational concepts you may have missed.")

    # 2. Risk-based general rules
    if risk_category in ["High Risk", "Moderate Risk"]:
        # If tutoring wasn't already suggested by the feature rules, add it as a safety net
        if not any("tutoring" in r.lower() for r in recommendations):
            recommendations.add("Given your current projection, actively seeking 1-on-1 tutoring or teacher office hours should be a priority.")
            
    # 3. Low Risk / Maintenance
    if risk_category == "Low Risk" and len(recommendations) == 0:
        recommendations.add("Excellent work! Keep maintaining your current study habits and attendance. Consistency is key to your continued success.")
    elif len(recommendations) == 0:
        # Fallback if no rules hit
        recommendations.add("Focus on maintaining a healthy, balanced routine of study, attendance, and self-care.")

    # Return top 5 unique recommendations
    return list(recommendations)[:5]