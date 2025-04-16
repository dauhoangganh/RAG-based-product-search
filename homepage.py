import streamlit as st

def main():
    st.logo(
        image="images/analog-logo_1709708907__17859.original.png",
        link="https://analog.com",
        size="large",
        icon_image="images/logo.jpg",
        )
    search_tool = st.Page("frontend_search_parts.py", title="Search tool", icon=":material/search:", default=True)
    cross_ref_search = st.Page("frontend_cross_search.py", title="Cross reference search", icon=":material/dashboard:")
    pg = st.navigation([search_tool, cross_ref_search])
    pg.run()


if __name__ == "__main__":
    main()
    