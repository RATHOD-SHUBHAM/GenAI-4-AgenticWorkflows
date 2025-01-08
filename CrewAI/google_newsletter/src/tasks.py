from textwrap import dedent
from crewai import Task
from google_newsletter.src.tools import search_tool

output_file_path = 'news_letter_output/output_v0.md'

class NewsLetterTasks:
    def news_collection_task(self, agent, topic, objective):
        return Task(
            description=dedent(f"""
            Your task is to gather the latest and most relevant news articles, reports, and insights based on the given topic and objective. 
            You will research a variety of reliable sources, including news outlets, blogs, press releases, and expert analyses. 
            Your goal is to collect up-to-date, accurate, and comprehensive news about the subject to provide a solid foundation for the newsletter.
    
            Topic: {topic}
            Objective: {objective}
            """),
            expected_output=dedent(f"""
            A comprehensive collection of the most relevant and up-to-date news articles, reports, and insights on the given topic. 
            The collected news should include varied perspectives from credible sources and offer a well-rounded understanding of the subject.
            """),
            name="news_collection_task",
            agent=agent,
            tools=[search_tool],
            async_execution=False
        )

    def news_critique_task(self, agent, topic, objective):
        return Task(
            description=dedent(f"""
            You are required to evaluate the collection of news gathered by the News Journalist. 
            Your job is to ensure that each article is relevant, accurate, and worth including in the newsletter. 
            Analyze the credibility of the sources, the timeliness of the news, and the overall quality of the content. 
            Remove any biased, outdated, or irrelevant information and identify key pieces that will contribute to the objective of the newsletter.

            Topic: {topic}
            Objective: {objective}
            """),
            expected_output=dedent(f"""
            A refined list of news articles and reports that meet the criteria of relevance, credibility, and timeliness. 
            This will include a recommendation for each article, noting whether it should be included, revised, or excluded from the newsletter based on its overall quality.
            """),
            name="news_critique_task",
            agent=agent,
            tools=[search_tool],
            async_execution=False
        )

    def newsletter_writing_task(self, agent, objective):
        return Task(
            description=dedent(f"""
            Based on the refined news content from the Critique agent, your task is to write a detailed and engaging newsletter. 
            This should include well-crafted, informative articles that are both accurate and engaging for the reader. 
            Pay attention to tone, structure, and clarity to ensure the final newsletter is appealing to the target audience. 
            Make sure the content aligns with the objective provided, and the stories are written in a way that makes them both interesting and informative.

            Objective: {objective}
            """),
            expected_output=dedent(f"""
            A fully written newsletter that includes a collection of articles based on the refined news. 
            The newsletter should be structured coherently, with engaging headlines, informative content, and a professional tone. 
            Each article should align with the newsletter’s goals and objectives, providing a valuable and enjoyable reading experience for the audience.
            """),
            name="newsletter_writing_task",
            agent=agent,
            async_execution=False,
            output_file=output_file_path
        )
