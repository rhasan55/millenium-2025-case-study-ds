import streamlit as st
import pandas as pd
from pathlib import Path
import os
import sys
import logging
from datetime import datetime
import json

import config
from ingestion import IngestionPipeline
from embeddings import HybridSearchEngine, EmbeddingEngine
from models import CandidateDatabase
from extraction import PIIExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Millennium Candidate Search",
    page_icon=str(config.LOGO_PATH) if config.LOGO_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_database():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set in environment")
        st.stop()

    pipeline = IngestionPipeline(api_key)
    database = pipeline.process_directory(config.RESUME_DIR)

    embedding_engine = EmbeddingEngine()
    search_engine = HybridSearchEngine(embedding_engine)

    return database, search_engine, pipeline

def safe_calculate_progression_score(candidate):
    """Safely calculate progression score with fallback for old cached objects"""
    try:
        if hasattr(candidate, 'calculate_career_progression_score'):
            return candidate.calculate_career_progression_score()
        else:
            # Fallback: return default structure
            return {
                "progression_velocity": 0.0,
                "pedigree_score": 0.0,
                "specialization_score": 0.0,
                "high_potential_indicators": [],
                "entry_level": "Unknown",
                "total_score": 0.0
            }
    except Exception as e:
        logger.error(f"Error calculating progression score: {e}")
        return {
            "progression_velocity": 0.0,
            "pedigree_score": 0.0,
            "specialization_score": 0.0,
            "high_potential_indicators": [],
            "entry_level": "Unknown",
            "total_score": 0.0
        }

def render_sidebar_filters():
    st.sidebar.header("Filters")

    filters = {}

    with st.sidebar.expander("Strategy & Asset Class", expanded=True):
        selected_strategies = st.multiselect(
            "Strategy",
            options=config.STRATEGIES,
            key="strategy_filter"
        )
        if selected_strategies:
            filters["strategy"] = selected_strategies

        selected_assets = st.multiselect(
            "Asset Class",
            options=config.ASSET_CLASSES,
            key="asset_filter"
        )
        if selected_assets:
            filters["asset_class"] = selected_assets

    with st.sidebar.expander("Sector Coverage", expanded=False):
        selected_sectors = st.multiselect(
            "Sectors",
            options=config.SECTORS,
            key="sector_filter"
        )
        if selected_sectors:
            filters["sector"] = selected_sectors

    with st.sidebar.expander("Experience & Location", expanded=False):
        exp_band = st.selectbox(
            "Experience Band",
            options=["All"] + config.EXPERIENCE_BANDS,
            key="exp_filter"
        )
        if exp_band != "All":
            filters["experience_band"] = exp_band

        location = st.text_input("Location", key="location_filter")
        if location:
            filters["location"] = location

        work_auth = st.selectbox(
            "Work Authorization",
            options=["All"] + config.WORK_AUTH_OPTIONS,
            key="auth_filter"
        )
        if work_auth != "All":
            filters["work_auth"] = work_auth

    with st.sidebar.expander("Education", expanded=False):
        school_tier = st.selectbox(
            "School Tier",
            options=["All", "Tier 1", "Tier 2"],
            key="school_filter"
        )
        if school_tier != "All":
            filters["school_tier"] = school_tier

        min_gpa = st.slider(
            "Minimum GPA",
            min_value=2.0,
            max_value=4.0,
            value=2.0,
            step=0.1,
            key="gpa_filter"
        )
        if min_gpa > 2.0:
            filters["min_gpa"] = min_gpa

    with st.sidebar.expander("Skills", expanded=False):
        prog_langs = st.multiselect(
            "Programming Languages",
            options=config.SKILLS_TAXONOMY["Programming"],
            key="prog_filter"
        )
        if prog_langs:
            filters["programming_langs"] = prog_langs

    return filters

def render_candidate_card(candidate, score, highlights, show_pii=False):
    with st.container():
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"### {candidate.name}")

            if candidate.location:
                st.caption(f"Location: {candidate.location}")

            if candidate.years_experience_band:
                st.caption(f"Experience: {candidate.years_experience_band} years")

        with col2:
            st.metric("Match Score", f"{score:.2f}")
            st.caption(f"Confidence: {candidate.meta.parse_confidence:.2f}")

            # Add progression score badge
            prog_score = safe_calculate_progression_score(candidate)
            total = prog_score["total_score"]
            if total >= 80:
                st.success(f"Elite: {total:.0f}")
            elif total >= 60:
                st.info(f"High-Potential: {total:.0f}")
            elif total >= 40:
                st.caption(f"Strong: {total:.0f}")
            else:
                st.caption(f"Progression: {total:.0f}")

        if candidate.strategy or candidate.asset_classes:
            strategy_text = " | ".join(candidate.strategy) if candidate.strategy else "N/A"
            asset_text = " | ".join(candidate.asset_classes) if candidate.asset_classes else "N/A"
            st.markdown(f"**Strategy:** {strategy_text} | **Asset Class:** {asset_text}")

        if candidate.sector_coverage:
            st.markdown(f"**Sectors:** {', '.join(candidate.sector_coverage[:5])}")

        if candidate.education:
            edu = candidate.education[0]
            edu_text = f"{edu.degree or 'Degree'} in {edu.major or 'Major'} from {edu.institution}"
            if edu.gpa:
                edu_text += f" (GPA: {edu.gpa})"
            st.markdown(f"Education: {edu_text}")

        if candidate.experience:
            exp = candidate.experience[0]
            st.markdown(f"Current: {exp.title} at {exp.employer}")

        if candidate.skills:
            top_skills = [s.skill for s in candidate.skills[:8]]
            st.markdown(f"**Skills:** {', '.join(top_skills)}")

        if highlights:
            with st.expander("Match Highlights"):
                for highlight in highlights:
                    st.markdown(f"- {highlight}")

        if show_pii:
            with st.expander("Contact Information"):
                if candidate.contacts.email:
                    if not candidate.contacts.email_is_valid:
                        st.warning(f"Email: {candidate.contacts.email} (incomplete/invalid)")
                    else:
                        st.text(f"Email: {candidate.contacts.email}")
                if candidate.contacts.phone:
                    st.text(f"Phone: {candidate.contacts.phone}")
                if candidate.contacts.linkedin:
                    st.text(f"LinkedIn: {candidate.contacts.linkedin}")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("View Full Profile", key=f"view_{candidate.candidate_id}"):
                st.session_state["selected_candidate"] = candidate.candidate_id
                st.session_state["page"] = "profile"
                st.rerun()

        with col2:
            if st.button("Add to Shortlist", key=f"short_{candidate.candidate_id}"):
                if "shortlist" not in st.session_state:
                    st.session_state["shortlist"] = []
                if candidate.candidate_id not in st.session_state["shortlist"]:
                    st.session_state["shortlist"].append(candidate.candidate_id)
                    st.success("Added to shortlist")

        st.markdown("---")

def search_page():
    st.title("Candidate Search")

    database, search_engine, pipeline = load_database()

    st.sidebar.markdown(f"**Total Candidates:** {len(database.candidates)}")

    filters = render_sidebar_filters()

    query = st.text_input(
        "Search candidates",
        placeholder="e.g., GLP-1, medtech device modeling, DCF",
        key="search_query"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.number_input("Results to show", min_value=5, max_value=100, value=20, step=5)
    with col2:
        show_pii = st.checkbox("Show PII", value=False)

    if st.button("Search", type="primary"):
        with st.spinner("Searching candidates..."):
            results = search_engine.search(
                query=query if query else None,
                candidates=database.candidates,
                candidate_embeddings=database.embeddings,
                filters=filters,
                top_k=top_k
            )

            st.session_state["search_results"] = results
            st.session_state["show_pii"] = show_pii

    if "search_results" in st.session_state and st.session_state["search_results"]:
        results = st.session_state["search_results"]
        show_pii = st.session_state.get("show_pii", False)

        st.markdown(f"### Found {len(results)} candidates")

        for candidate, score, highlights in results:
            render_candidate_card(candidate, score, highlights, show_pii)

    else:
        st.info("Use filters and search to find candidates")

def profile_page():
    database, _, _ = load_database()

    if "selected_candidate" not in st.session_state:
        st.error("No candidate selected")
        if st.button("Back to Search"):
            st.session_state["page"] = "search"
            st.rerun()
        return

    candidate_id = st.session_state["selected_candidate"]
    candidate = database.get_by_id(candidate_id)

    if not candidate:
        st.error("Candidate not found")
        return

    if st.button("← Back to Search"):
        st.session_state["page"] = "search"
        st.rerun()

    st.title(candidate.name)

    # Career Progression Score prominently displayed
    prog_data = safe_calculate_progression_score(candidate)
    total_score = prog_data["total_score"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Progression Score", f"{total_score:.0f}/100")
        if total_score >= 80:
            st.caption("Elite Candidate")
        elif total_score >= 60:
            st.caption("High-Potential")
        elif total_score >= 40:
            st.caption("Strong Profile")

    with col2:
        if candidate.location:
            st.metric("Location", candidate.location)
    with col3:
        if candidate.years_experience_band:
            st.metric("Experience", candidate.years_experience_band)
    with col4:
        entry_level = prog_data["entry_level"]
        st.metric("Entry Level", entry_level)

    # Progression score breakdown
    if prog_data["high_potential_indicators"]:
        with st.expander("High-Potential Indicators", expanded=True):
            for indicator in prog_data["high_potential_indicators"]:
                st.markdown(f"- {indicator}")

    # Score components
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pedigree", f"{prog_data['pedigree_score']:.0f}/40", help="Top school + GPA + Awards")
    with col2:
        st.metric("Velocity", f"{prog_data['progression_velocity']:.0f}/35", help="Promotions + Firm quality")
    with col3:
        st.metric("Specialization", f"{prog_data['specialization_score']:.0f}/25", help="Sector depth + Research")

    st.markdown("### Contact Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        if candidate.contacts.email:
            if not candidate.contacts.email_is_valid:
                st.warning(f"Email: {candidate.contacts.email}")
                st.caption("Incomplete or invalid email address")
            else:
                st.text(f"Email: {candidate.contacts.email}")
    with col2:
        if candidate.contacts.phone:
            st.text(f"Phone: {candidate.contacts.phone}")
    with col3:
        if candidate.contacts.linkedin:
            st.markdown(f"[LinkedIn]({candidate.contacts.linkedin})")

    if candidate.strategy or candidate.asset_classes or candidate.sector_coverage:
        st.markdown("### Investment Profile")
        col1, col2 = st.columns(2)
        with col1:
            if candidate.strategy:
                st.markdown(f"**Strategy:** {', '.join(candidate.strategy)}")
            if candidate.asset_classes:
                st.markdown(f"**Asset Classes:** {', '.join(candidate.asset_classes)}")
        with col2:
            if candidate.sector_coverage:
                st.markdown(f"**Sectors:** {', '.join(candidate.sector_coverage)}")

    if candidate.education:
        st.markdown("### Education")
        for edu in candidate.education:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{edu.institution}**")
                    if edu.degree or edu.major:
                        st.text(f"{edu.degree or ''} {edu.major or ''}")
                with col2:
                    if edu.gpa:
                        st.metric("GPA", edu.gpa)
                if edu.honors:
                    st.caption(f"Honors: {', '.join(edu.honors)}")
                st.markdown("---")

    if candidate.experience:
        st.markdown("### Experience")
        for exp in candidate.experience:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{exp.title}**")
                    st.text(exp.employer)
                with col2:
                    date_range = f"{exp.start_date or 'N/A'} - {exp.end_date or 'Present'}"
                    st.caption(date_range)

                if exp.description:
                    st.text(exp.description[:300])

                if exp.sector_coverage or exp.asset_class or exp.strategy:
                    badges = []
                    if exp.sector_coverage:
                        badges.extend(exp.sector_coverage)
                    if exp.asset_class:
                        badges.extend(exp.asset_class)
                    if exp.strategy:
                        badges.extend(exp.strategy)
                    st.caption(" | ".join(badges))

                st.markdown("---")

    if candidate.skills:
        st.markdown("### Skills")
        skill_families = {}
        for skill in candidate.skills:
            family = skill.family or "Other"
            if family not in skill_families:
                skill_families[family] = []
            skill_families[family].append(skill.skill)

        for family, skills in skill_families.items():
            with st.expander(f"{family} ({len(skills)})"):
                st.markdown(", ".join(skills))

    if candidate.projects:
        st.markdown("### Projects")
        for proj in candidate.projects:
            with st.expander(proj.name):
                if proj.description:
                    st.text(proj.description)
                if proj.technologies:
                    st.caption(f"Technologies: {', '.join(proj.technologies)}")

    if candidate.publications:
        st.markdown("### Publications")
        for pub in candidate.publications:
            st.markdown(f"- **{pub.title}**" + (f" ({pub.venue})" if pub.venue else ""))

    if candidate.certifications:
        st.markdown("### Certifications")
        for cert in candidate.certifications:
            st.markdown(f"- {cert}")

    st.markdown("### Metadata")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Parse Confidence", f"{candidate.meta.parse_confidence:.2%}")
    with col2:
        if candidate.meta.processing_time_ms:
            st.metric("Processing Time", f"{candidate.meta.processing_time_ms:.0f}ms")
    with col3:
        st.metric("Source File", candidate.meta.source_file)

def insights_page():
    st.title("Pipeline Insights")

    database, _, pipeline = load_database()

    stats = pipeline.get_processing_stats(database)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Candidates", stats["total_candidates"])
    with col2:
        st.metric("Avg Confidence", f"{stats['avg_parse_confidence']:.2%}")
    with col3:
        st.metric("Avg Processing Time", f"{stats['avg_processing_time_ms']:.0f}ms")
    with col4:
        unique_schools = len(set([edu.institution for c in database.candidates for edu in c.education]))
        st.metric("Unique Schools", unique_schools)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Strategy Distribution")
        if stats["strategies"]:
            strategy_df = pd.DataFrame([
                {"Strategy": k, "Count": v}
                for k, v in stats["strategies"].items()
            ])
            st.bar_chart(strategy_df.set_index("Strategy"))
        else:
            st.info("No strategy data")

    with col2:
        st.markdown("### Top Sectors")
        if stats["sectors"]:
            sorted_sectors = sorted(stats["sectors"].items(), key=lambda x: x[1], reverse=True)[:10]
            sector_df = pd.DataFrame([
                {"Sector": k, "Count": v}
                for k, v in sorted_sectors
            ])
            st.bar_chart(sector_df.set_index("Sector"))
        else:
            st.info("No sector data")

    st.markdown("### Region Distribution")
    region_counts = {"North America": 0, "EMEA": 0, "APAC": 0, "Other": 0}
    for candidate in database.candidates:
        if candidate.location:
            loc = candidate.location.lower()
            if any(state.lower() in loc for state in ["ny", "ca", "tx", "il", "usa", "us", "united states"]):
                region_counts["North America"] += 1
            elif any(country in loc for country in ["uk", "london", "paris", "germany", "europe"]):
                region_counts["EMEA"] += 1
            elif any(country in loc for country in ["singapore", "hong kong", "tokyo", "asia"]):
                region_counts["APAC"] += 1
            else:
                region_counts["Other"] += 1

    region_df = pd.DataFrame([
        {"Region": k, "Count": v}
        for k, v in region_counts.items() if v > 0
    ])
    if not region_df.empty:
        st.bar_chart(region_df.set_index("Region"))
    else:
        st.info("No region data available")

    st.markdown("### Career Progression Intelligence")
    st.caption("Analyzing career trajectories, pedigree, and high-potential signals")

    # Calculate progression scores for all candidates
    progression_data = []
    for candidate in database.candidates:
        score_data = safe_calculate_progression_score(candidate)
        progression_data.append({
            "name": candidate.name,
            "total_score": score_data["total_score"],
            "pedigree": score_data["pedigree_score"],
            "velocity": score_data["progression_velocity"],
            "specialization": score_data["specialization_score"],
            "entry_level": score_data["entry_level"],
            "indicators": score_data["high_potential_indicators"]
        })

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Progression Score Distribution")
        score_ranges = {
            "Elite (80-100)": 0,
            "High Potential (60-79)": 0,
            "Strong (40-59)": 0,
            "Developing (20-39)": 0,
            "Entry (<20)": 0
        }

        for data in progression_data:
            score = data["total_score"]
            if score >= 80:
                score_ranges["Elite (80-100)"] += 1
            elif score >= 60:
                score_ranges["High Potential (60-79)"] += 1
            elif score >= 40:
                score_ranges["Strong (40-59)"] += 1
            elif score >= 20:
                score_ranges["Developing (20-39)"] += 1
            else:
                score_ranges["Entry (<20)"] += 1

        score_df = pd.DataFrame([
            {"Tier": k, "Count": v}
            for k, v in score_ranges.items() if v > 0
        ])
        if not score_df.empty:
            st.bar_chart(score_df.set_index("Tier"))
        else:
            st.info("No progression data")

    with col2:
        st.markdown("#### Entry Level Distribution")
        entry_level_counts = {}
        for data in progression_data:
            level = data["entry_level"]
            entry_level_counts[level] = entry_level_counts.get(level, 0) + 1

        entry_df = pd.DataFrame([
            {"Entry Level": k, "Count": v}
            for k, v in entry_level_counts.items()
        ])
        if not entry_df.empty:
            st.bar_chart(entry_df.set_index("Entry Level"))
        else:
            st.info("No entry level data")

    # Top performers table
    st.markdown("#### Top Performers by Progression Score")
    top_performers = sorted(progression_data, key=lambda x: x["total_score"], reverse=True)[:10]

    top_performers_display = []
    for performer in top_performers:
        indicators_str = ", ".join(performer["indicators"][:3]) if performer["indicators"] else "N/A"
        top_performers_display.append({
            "Name": performer["name"],
            "Total Score": f"{performer['total_score']:.0f}",
            "Pedigree": f"{performer['pedigree']:.0f}",
            "Velocity": f"{performer['velocity']:.0f}",
            "Specialization": f"{performer['specialization']:.0f}",
            "Key Indicators": indicators_str
        })

    if top_performers_display:
        st.dataframe(pd.DataFrame(top_performers_display), use_container_width=True, hide_index=True)

    # Score component breakdown
    st.markdown("#### Score Component Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_pedigree = sum(d["pedigree"] for d in progression_data) / len(progression_data) if progression_data else 0
        st.metric("Avg Pedigree Score", f"{avg_pedigree:.1f}/40")
        st.caption("Education + GPA + Awards")

    with col2:
        avg_velocity = sum(d["velocity"] for d in progression_data) / len(progression_data) if progression_data else 0
        st.metric("Avg Velocity Score", f"{avg_velocity:.1f}/35")
        st.caption("Promotions + Firm Quality")

    with col3:
        avg_spec = sum(d["specialization"] for d in progression_data) / len(progression_data) if progression_data else 0
        st.metric("Avg Specialization Score", f"{avg_spec:.1f}/25")
        st.caption("Sector Depth + Research")

    # High-potential indicators frequency
    st.markdown("#### Most Common High-Potential Indicators")
    indicator_counts = {}
    for data in progression_data:
        for indicator in data["indicators"]:
            indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1

    if indicator_counts:
        sorted_indicators = sorted(indicator_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        indicator_df = pd.DataFrame([
            {"Indicator": k, "Count": v}
            for k, v in sorted_indicators
        ])
        st.dataframe(indicator_df, use_container_width=True, hide_index=True)

    st.markdown("### Top Skills")
    skill_counts = {}
    for candidate in database.candidates:
        for skill in candidate.skills:
            skill_counts[skill.skill] = skill_counts.get(skill.skill, 0) + 1

    sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    skill_df = pd.DataFrame([
        {"Skill": k, "Count": v}
        for k, v in sorted_skills
    ])
    if not skill_df.empty:
        st.bar_chart(skill_df.set_index("Skill"))
    else:
        st.info("No skills data available")

def export_page():
    st.title("Export Candidates")

    database, _, _ = load_database()

    export_type = st.radio("Export Type", ["All Candidates", "Search Results", "Shortlist"])

    include_pii = st.checkbox("Include PII (Email, Phone, LinkedIn)", value=False)

    candidates_to_export = []

    if export_type == "All Candidates":
        candidates_to_export = database.candidates
    elif export_type == "Search Results":
        if "search_results" in st.session_state:
            candidates_to_export = [c for c, _, _ in st.session_state["search_results"]]
        else:
            st.warning("No search results available")
    elif export_type == "Shortlist":
        if "shortlist" in st.session_state:
            shortlist_ids = st.session_state["shortlist"]
            candidates_to_export = [database.get_by_id(cid) for cid in shortlist_ids if database.get_by_id(cid)]
        else:
            st.warning("Shortlist is empty")

    if candidates_to_export:
        st.info(f"Ready to export {len(candidates_to_export)} candidates")

        export_data = []
        for candidate in candidates_to_export:
            row = {
                "Name": candidate.name,
                "Location": candidate.location or "",
                "Work Auth": candidate.work_auth or "",
                "Experience Band": candidate.years_experience_band or "",
                "Strategy": ", ".join(candidate.strategy),
                "Asset Classes": ", ".join(candidate.asset_classes),
                "Sectors": ", ".join(candidate.sector_coverage),
                "Skills": ", ".join([s.skill for s in candidate.skills[:10]]),
                "School": candidate.education[0].institution if candidate.education else "",
                "Degree": candidate.education[0].degree if candidate.education else "",
                "GPA": candidate.education[0].gpa if candidate.education else "",
                "Current Title": candidate.experience[0].title if candidate.experience else "",
                "Current Employer": candidate.experience[0].employer if candidate.experience else "",
                "Parse Confidence": candidate.meta.parse_confidence
            }

            if include_pii:
                row["Email"] = candidate.contacts.email or ""
                row["Phone"] = candidate.contacts.phone or ""
                row["LinkedIn"] = candidate.contacts.linkedin or ""
            else:
                row["Email"] = "***REDACTED***" if candidate.contacts.email else ""
                row["Phone"] = "***REDACTED***" if candidate.contacts.phone else ""
                row["LinkedIn"] = "***REDACTED***" if candidate.contacts.linkedin else ""

            export_data.append(row)

        df = pd.DataFrame(export_data)

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"candidates_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        json_data = [candidate.to_dict_with_pii_control(include_pii) for candidate in candidates_to_export]
        json_str = json.dumps(json_data, indent=2, default=str)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"candidates_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    if config.LOGO_PATH.exists():
        st.sidebar.image(str(config.LOGO_PATH), width=100)

    st.sidebar.title("Navigation")

    # Add cache clear button
    if st.sidebar.button("Clear Cache & Reload Data"):
        st.cache_resource.clear()
        st.rerun()
    pages = {
        "Search": search_page,
        "Insights": insights_page,
        "Export": export_page
    }

    if "page" not in st.session_state:
        st.session_state["page"] = "search"

    if st.session_state.get("page") == "profile":
        profile_page()
    else:
        selection = st.sidebar.radio("Go to", list(pages.keys()), index=list(pages.keys()).index(st.session_state.get("page", "Search").title()) if st.session_state.get("page", "search").title() in pages else 0)

        st.session_state["page"] = selection.lower()

        pages[selection]()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    main()
