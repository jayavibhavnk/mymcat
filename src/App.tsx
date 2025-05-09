import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import { ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Document } from "langchain/document";

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'bot';
}

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    { id: Date.now(), text: 'Hello! I am a RAG-powered chatbot. You can ask me questions about the MCAT content outline.', sender: 'bot' }
  ]);
  const [input, setInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [vectorStore, setVectorStore] = useState<MemoryVectorStore | null>(null);
  const [apiKey, setApiKey] = useState<string>('');
  const [isApiKeySet, setIsApiKeySet] = useState<boolean>(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleApiKeySubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (apiKey.trim() !== '') {
      setIsApiKeySet(true);
    }
  };

const textfi = `®

What’s on the MCAT Exam?

students-residents.aamc.org/mcatexam

MCAT® is a program of the
Association of American Medical Colleges
www.aamc.org/mcat

Contents
Introduction .................................................................................................................................................. 2
Scientific Inquiry and Reasoning Skills .......................................................................................................... 4
Biological and Biochemical Foundations of Living Systems ........................................................................ 17
Chemical and Physical Foundations of Biological Systems ......................................................................... 50
Psychological, Social, and Biological Foundations of Behavior................................................................... 75
Critical Analysis and Reasoning Skills ........................................................................................................ 101

© 2020 Association of American Medical Colleges

1

Introduction
This document draws from the online resource What’s on the MCAT® Exam? at studentsresidents.aamc.org/mcatexam. It contains a complete description of the competencies you are
responsible for knowing on the MCAT exam. It describes the exam’s content and format. It also lists and
discusses the exam’s conceptual framework, organized around foundational concepts, content
categories, and scientific inquiry and reasoning skills. Also included are sample test questions that
demonstrate how the competencies are tested on the exam. While the content is written for you, the
prospective MCAT examinee, the information it provides is likely to be useful to prehealth advisors,
other baccalaureate faculty, medical school admissions officers, and medical schools.

How Is the MCAT Exam Structured?
The MCAT exam has four test sections:
▪
▪
▪
▪

Biological and Biochemical Foundations of Living Systems
Chemical and Physical Foundations of Biological Systems
Psychological, Social, and Biological Foundations of Behavior
Critical Analysis and Reasoning Skills

The first three sections are organized around foundational concepts, or “big ideas,” in the sciences. They
reflect current research about the most effective ways for students to learn and use science,
emphasizing deep knowledge of the most important scientific concepts over knowledge simply of many
discrete scientific facts.
Leaders in science education say some of the most important foundational concepts in the sciences ask
students to integrate and analyze information from different disciplines. In that vein, questions in these
sections will ask you to combine your scientific knowledge from multiple disciplines with your scientific
inquiry and reasoning skills. You will be asked to demonstrate four different scientific inquiry and
reasoning skills on the exam:
▪
▪
▪
▪

Knowledge of scientific concepts and principles.
Scientific reasoning and problem-solving.
Reasoning about the design and execution of research.
Data-based and statistical reasoning.

The fourth section of the MCAT exam, Critical Analysis and Reasoning Skills, will be similar to many of
the verbal reasoning tests you have taken in your academic career. It includes passages and questions
that test your ability to comprehend and analyze what you read. The Critical Analysis and Reasoning
Skills section asks you to read and think about passages from a wide range of disciplines in the social
sciences and humanities, including those in population health, ethics and philosophy, and studies of
diverse cultures. Passages are followed by a series of questions that lead you through the process of

© 2020 Association of American Medical Colleges

2

comprehending, analyzing, and reasoning about the material you have read. This section is unique
because it has been developed specifically to measure the analytical and reasoning skills you will need to
be successful in medical school.

© 2020 Association of American Medical Colleges

3

Scientific Inquiry and Reasoning Skills
Leaders in medical education believe tomorrow’s physicians need to be able to combine scientific
knowledge with skills in scientific inquiry and reasoning. With that in mind, the MCAT exam will ask you
to demonstrate four scientific inquiry and reasoning skills that natural, behavioral, and social scientists
rely on to advance their work:
Knowledge of Scientific Concepts and Principles
• Demonstrating understanding of scientific concepts and principles.
• Identifying the relationships between closely related concepts.

Scientific Reasoning and Problem-Solving
• Reasoning about scientific principles, theories, and models.
• Analyzing and evaluating scientific explanations and predictions.

Reasoning About the Design and Execution of Research
• Demonstrating understanding of important components of scientific research.
• Reasoning about ethical issues in research.

Data-Based and Statistical Reasoning
• Interpreting patterns in data presented in tables, figures, and graphs.
• Reasoning about data and drawing conclusions from them.

The discussion that follows describes each of the skills and how you may be asked to demonstrate them.
Three sample test questions are provided to illustrate each skill: one from the Psychological, Social, and
Biological Foundations of Behavior section; one from the Biological and Biochemical Foundations of
Living Systems section; and one from the Chemical and Physical Foundations of Biological Systems
section. Also included are explanations of how each question tests a specific scientific inquiry and
reasoning skill.

© 2020 Association of American Medical Colleges

4

Skill 1: Knowledge of Scientific Concepts and Principles
The questions in this skill category will ask you to demonstrate your knowledge of the 10 foundational
concepts described in subsequent chapters. These questions will ask you to recognize, identify, recall, or
define basic concepts in the natural, behavioral, and social sciences as well as their relationships with
one another. The concepts and scientific principles may be represented by words, graphs, tables,
diagrams, or formulas.
As you work on these questions, you may be asked to identify a scientific fact or define a concept. Or
you may be asked to apply a scientific principle to a problem. Questions may ask you to identify the
relationships between closely related concepts or relate written statements, principles, or concepts to
graphic representations of science content. They may ask you to identify examples of natural or datadriven observations that illustrate scientific principles. Questions may ask you to recognize a scientific
concept shown in a diagram or represented in a graph.
Or they may give you a mathematical equation and ask you to use it to solve a problem.
For example, questions that test this skill will ask you to show you understand scientific concepts and
principles by:
▪
▪
▪
▪
▪

Recognizing scientific principles from an example, situation, or study. Identifying the
relationships among closely related concepts.
Identifying the relationships between different representations of concepts (e.g., written,
symbolic, graphic).
Identifying examples of observations that illustrate scientific principles.
Using given mathematical equations to solve problems.
Identifying the simple or familiar molecule that is an example of a specific amino acid.

By way of example, questions from the Psychological, Social, and Biological Foundations of Behavior
section may ask you to demonstrate your knowledge of scientific concepts and principles by:
▪
▪
▪
▪
▪
▪
▪

Recognizing the principle of retroactive interference.
Using Weber’s law to identify physical differences that are detectable.
Identifying the behavioral change (extinction) that will occur when a learned response is no
longer followed by a reinforcer.
Identifying the conceptual similarities or differences between operant conditioning and classical
conditioning.
Identifying a graph that illustrates the relationship between educational attainment and life
expectancy.
Recognizing conditions that result in learned helplessness.
Concluding which stage of cognitive development a child is in, according to Piaget’s theory,
when presented with a description of how a child responds to a conservation problem.

© 2020 Association of American Medical Colleges

5

The three sample questions that follow illustrate Skill 1 questions from, respectively, the Psychological,
Social, and Biological Foundations of Behavior section; the Biological and Biochemical Foundations of
Living Systems section; and the Chemical and Physical Foundations of Biological System section of the
MCAT exam.
Skill 1 Example From the Psychological, Social, and Biological Foundations of Behavior Section
In a study, each trial involves administering a drop of lemon juice to the participant’s tongue and
measuring the participant’s level of salivation. As more trials are conducted, the researcher finds that
the magnitude of salivation declines. After a certain point, the researcher switches to administering lime
juice. This researcher is most likely studying which process?
A.
B.
C.
D.

Sensory perception
Habituation and dishabituation
Stimulus generalization in classical conditioning
Conditioned responses in classical conditioning

The correct answer is B. This Skill 1 question tests your knowledge of the scientific concepts and
principles described by Content Category 7C, Attitude and behavior change (see page 90), and is a Skill 1
question because it requires you to relate scientific concepts. This question asks you to identify the
process involved in the study that connects reduced responding to a repeated stimulus and then a
change in the stimulus, which is habituation and dishabituation, allowing for the conclusion that B is the
correct answer.
Skill 1 Example From the Chemical and Physical Foundations of Biological Systems Section
What type of functional group is formed when aspartic acid reacts with another amino acid to form a
peptide bond?
A.
B.
C.
D.

An amine group
An aldehyde group
An amide group
A carboxyl group

The correct answer is C. This is a Skill 1 question and relates to Content Category 5D, Structure, function,
and reactivity of biologically relevant molecules. It is a Skill 1 question because you must recognize the
structural relationship between free amino acids and peptides. To answer the question, you must know
that the functional group that forms during peptide bond formation is an amide group.

Skill 2: Scientific Reasoning and Problem-Solving
Questions that test scientific reasoning and problem-solving skills differ from questions in the previous
category by asking you to use your scientific knowledge to solve problems in the natural, behavioral, and
social sciences.

© 2020 Association of American Medical Colleges

6

As you work on questions that test this skill, you may be asked to use scientific theories to explain
observations or make predictions about natural or social phenomena. Questions may ask you to judge
the credibility of scientific explanations or to evaluate arguments about cause and effect. Or they may
ask you to use scientific models and observations to draw conclusions. They may ask you to identify
scientific findings that call a theory or model into question. Questions in this category may ask you to
look at pictures or diagrams and draw conclusions from them. Or they may ask you to determine and
then use scientific formulas to solve problems.
For example, you will be asked to show you can use scientific principles to solve problems by:
▪
▪
▪
▪
▪
▪
▪

Reasoning about scientific principles, theories, and models to make predictions or determine
consequences.
Analyzing and evaluating the validity or credibility of scientific explanations and predictions.
Evaluating arguments about causes and consequences to determine the most valid argument
when using scientific knowledge.
Bringing together theory, observations, and evidence to draw conclusions.
Recognizing or identifying scientific findings from a given study that challenge or invalidate a
scientific theory or model.
Determining and using scientific formulas to solve problems.
Identifying the bond that would form between two structures if they were adjacent to each
other.

By way of illustration, questions from the Psychological, Social, and Biological Foundations of Behavior
section may ask you demonstrate this skill by:
▪

▪
▪
▪
▪

▪
▪

Using the main premises of symbolic interactionism, use reasoning in an observational study of
physician-patient interactions to describe how the premises are connected to perceived patient
compliance.
Predicting how an individual will react to cognitive dissonance.
Using reasoning to determine whether a causal explanation is possible when given an example
of how someone’s gender or personality predicts his or her behavior.
Explaining how an example, such as when an anorexic teenager restricts eating to satisfy esteem
needs, is compatible with the premises of Maslow’s hierarchy of needs.
Drawing a conclusion about which sociological theory would be most consistent with a
conceptual diagram that explains how social and environmental factors influence health and
why this theory is most consistent.
Identifying the relationship between social institutions that is suggested by an illustration used
in a public health campaign.
Recognizing a demographic trend that is represented in a population pyramid.

© 2020 Association of American Medical Colleges

7

For more context, let’s consider three Skill 2 questions linked to different foundational concepts in the
Psychological, Social, and Biological Foundations of Behavior section; the Biological and Biochemical
Foundations of Living Systems section; and the Chemical and Physical Foundations of Biological Systems
section.
Skill 2 Example From the Psychological, Social, and Biological Foundations of Behavior Section
Which statement describes what the concept of cultural capital predicts?
A. Cultural distinctions associated with the young will be more valued within a society.
B. With improved communication, there will eventually be a convergence of cultural practices of
all classes.
C. Cultural distinctions by class will become less important during a recession because people will
have less money to spend.
D. Cultural distinctions associated with elite classes will be more valued within a society.
The correct answer is D. It is a Skill 2 question and assesses knowledge of Content Category 10A, Social
inequality. It is a Skill 2 question because it requires you to make a prediction based on a particular
concept. This question requires you to understand the concept of cultural capital in order to evaluate
which prediction about social stratification would be most consistent with the concept.
Skill 2 Example From the Biological and Biochemical Foundations of Living Systems Section
Starting with the translation initiation codon, how many amino acids for this polypeptide does the
sequence shown encode?
5'-CUGCCAAUGUGCUAAUCGCGGGGG-3'
A.
B.
C.
D.

2
3
6
8

The correct answer is A. This is a Skill 2 question, and you must use knowledge from Content Category
1B, Transmission of genetic information from the gene to the protein, to solve this problem. In addition
to recalling the sequence for the start codon, this is a Skill 2 question because it requires you to apply
the scientific principle of the genetic code to the provided RNA sequence. As a Skill 2 question,
reasoning about the role of the stop codon in translation will allow you to arrive at the conclusion that
this sequence codes for a polypeptide with two amino acids.

© 2020 Association of American Medical Colleges

8

Skill 2 Example From the Chemical and Physical Foundations of Biological Systems Section
The radius of the aorta is about 1.0 cm, and blood passes through it at a velocity of 30 cm/s. A typical
capillary has a radius of about 4 × 10–4 cm, with blood passing through at a velocity of 5 × 10–2 cm/s.
Using these data, what is the approximate number of capillaries in a human body?
A. 1 × 104
B. 2 × 107
C. 4 × 109
D. 7 × 1012
The correct answer is C. This Skill 2 question relates to Content Category 4B, Importance of fluids for the
circulation of blood, gas movement, and gas exchange. This question asks you to use a mathematical
model to make predictions about natural phenomena. To answer this question, you must be able to
recognize the principles of flow characteristics of blood in the human body and apply the appropriate
mathematical model to an unfamiliar scenario. Answering this question first requires recognition that
the volume of blood flowing through the aorta is the same volume of blood flowing through the
capillaries. It is a Skill 2 question because you then need to use reasoning skills to find the difference in
the volumes that the aorta and capillaries can each carry in order to calculate the total number of
capillaries.

Skill 3: Reasoning About the Design and Execution of Research
Questions that test reasoning about the design and execution of research will ask you to demonstrate
your scientific inquiry skills by showing you can “do” science. They will ask you to demonstrate your
understanding of important components of scientific methodology. These questions will ask you to
demonstrate your knowledge of the ways natural, behavioral, and social scientists conduct research to
test and extend scientific knowledge.
As you work on these questions, you may be asked to show how scientists use theory, past research
findings, and observations to ask testable questions and pose hypotheses. Questions that test this skill
may ask you to use reasoning to identify the best way for scientists to gather data from samples of
members of the population they would like to draw inferences about. They may ask you to identify how
scientists manipulate and control variables to test their hypotheses or to identify and determine
different ways scientists take measurements and record results. The questions may ask you to identify
faulty research logic or point out the limitations of the research studies that are described. Or they may
ask you to identify factors that might confuse or confound the inferences you can draw from the results.
These questions may also ask you to demonstrate and use your understanding of the ways scientists
adhere to ethical guidelines to protect the rights, safety, and privacy of research participants, the
integrity of the scientists’ work, and the interests of research consumers.

© 2020 Association of American Medical Colleges

9

For example, questions that test this skill will ask you to use your knowledge of important components
of scientific methodology by:
▪
▪
▪
▪
▪
▪
▪
▪
▪

Identifying the role of theory, past findings, and observations in scientific questioning.
Identifying testable research questions and hypotheses.
Distinguishing between samples and populations and between results that support and fail to
support generalizations about populations.
Identifying the relationships among the variables in a study (e.g., independent versus dependent
variables; control and confounding variables).
Using reasoning to evaluate the appropriateness, precision, and accuracy of tools used to
conduct research in the natural sciences.
Using reasoning to evaluate or determine the appropriateness, reliability, and validity of tools
used to conduct research in the behavioral and social sciences.
Using reasoning to determine which features of research studies suggest associations between
variables or causal relationships between them (e.g., temporality, random assignment).
Using reasoning to evaluate ethical issues when given information about a study.
Determining which molecule is a product of two other molecules without rearrangement.

For example, questions from the Psychological, Social, and Biological Foundations of Behavior section
may ask you to reason about the design and execution of research by:
▪
▪
▪
▪
▪
▪
▪
▪
▪

Identifying the basic components of survey methods, ethnographic methods, experimental
methods, or other types of research designs in psychology and sociology.
Selecting a hypothesis about semantic activation.
Identifying the extent to which a finding can be generalized to the population when given details
about how participants were recruited for an experiment in language development.
Identifying the experimental setup in which researchers manipulate self-confidence.
Identifying the most appropriate way to assess prejudice in a study on implicit bias.
Using reasoning to determine or evaluate the implications of relying on self-report measures for
a specific study.
Identifying the third variable that may be confounding the findings from a correlational study.
Making judgments about the reliability and validity of specific measures when given information
about the response patterns of participants.
Identifying whether researchers violated any ethical codes when given information about a
study.

The three sample questions that follow illustrate Skill 3 questions from, respectively, the Psychological,
Social, and Biological Foundations of Behavior section; the Biological and Biochemical Foundations of
Living Systems section; and the Chemical and Physical Foundations of Biological Systems section of the
MCAT exam.

© 2020 Association of American Medical Colleges

10

Skill 3 Example From the Psychological, Social, and Biological Foundations of Behavior Section
Researchers conducted an experiment to test social loafing. They asked participants to prepare an
annual report or a tax return. Some participants performed the task individually and others performed it
as a group. What are the independent and dependent variables?
A. The independent variable is the overall productivity of the group, and the dependent variable is
each participant’s contribution to the task.
B. The independent variable is the type of task, and the dependent variable is whether the
participants worked alone or in a group.
C. The independent variable is whether the participant worked alone or in a group, and the
dependent variable is each participant’s contribution to the task.
D. The independent variable is whether the participant worked alone or in a group, and the
dependent variable is the type of the task.
The correct answer is C. This Skill 3 question assesses knowledge of Content Category 7B, Social
processes that influence human behavior. This question is a Skill 3 question because it requires you to
use reasoning skills in research design. This question requires you to understand social loafing and draw
inferences about the dependent and independent variables based on this concept and the description of
the experimental design.
Skill 3 Example from the Biological and Biochemical Foundations of Living Systems Section
Sodium dodecyl sulfate (SDS) contains a 12-carbon tail attached to a sulfate group and is used in
denaturing gel electrophoresis of proteins. Numerous SDS molecules will bind to the exposed
hydrophobic regions of denatured proteins. How does the use of SDS in this experiment allow for the
separation of proteins?
A. by charge
B. by molecular weight
C. by shape
D. by solubility
The correct answer is B. This is a Skill 3 question and requires knowledge from Content Category 1A,
Structure and function of proteins and their constituent amino acids. It is a Skill 3 question because it
requires you to understand the design of a denaturing gel electrophoresis experiment and the role that
SDS plays in this technique. Based on this understanding, you will be able to determine that proteins will
be separated only by molecular weight.

© 2020 Association of American Medical Colleges

11

Skill 3 Example From the Chemical and Physical Foundations of Biological Systems Section
A test for proteins in urine involves precipitation but is often complicated by precipitation of calcium
phosphate. Which procedure prevents precipitation of the salt?
A. addition of buffer to maintain high pH
B. addition of buffer to maintain neutral pH
C. addition of calcium hydroxide
D. addition of sodium phosphate
The correct answer is B. This is a Skill 3 question and relates to Content Category 5B, Nature of
molecules and intermolecular interactions. In this Skill 3 question, you must identify a change in an
experimental approach that would eliminate a frequently encountered complication. The complication
in this case is related to the test for protein-involving precipitation. The test will give a false positive if
calcium phosphate precipitates. To answer this Skill 3 question, you need to use reasoning skills to
determine how changing experimental parameters will eliminate the complication.

Skill 4: Data-Based and Statistical Reasoning
Like questions about Skill 3, questions that test Skill 4 will ask you to show you can “do” science, this
time by demonstrating your data-based and statistical reasoning skills. Questions that test this skill will
ask you to reason with data. They will ask you to read and interpret results using tables, graphs, and
charts. These questions will ask you to demonstrate you can identify patterns in data and draw
conclusions from evidence.
Questions that test this skill may ask you to demonstrate your knowledge of the ways natural,
behavioral, and social scientists use measures of central tendency and dispersion to describe their data.
These questions may ask you to demonstrate your understanding of the ways scientists think about
random and systematic errors in their experiments and datasets. They may also ask you to demonstrate
your understanding of how scientists think about uncertainty and the implications of uncertainty for
statistical testing and the inferences they can draw from their data. These questions may ask you to
show how scientists use data to make comparisons between variables or explain relationships between
them or make predictions. They may ask you to use data to answer research questions or draw
conclusions.
These questions may ask you to demonstrate your knowledge of the ways scientists draw inferences
from their results about associations between variables or causal relationships between them.
Questions that test this skill may ask you to examine evidence from a scientific study and point out
statements that go beyond the evidence. Or they may ask you to suggest alternative explanations for
the same data.

© 2020 Association of American Medical Colleges

12

For example, questions that test this skill will ask you to use your knowledge of data-based and
statistical reasoning by:
▪
▪
▪
▪
▪

▪
▪
▪
▪
▪

Using, analyzing, and interpreting data in figures, graphs, and tables to draw a conclusion about
expected results if the experiment was to be completed again.
Evaluating whether representations are an appropriate or reliable fit for particular scientific
observations and data.
Using measures of central tendency (mean, median, and mode) and measures of dispersion
(range, inter-quartile range, and standard deviation) to describe data.
Using reasoning about random and systematic error.
Using reasoning about statistical significance and uncertainty (e.g., interpreting statistical
significance levels, interpreting a confidence interval) and relating this information to
conclusions that can or cannot be made about the study.
Using data to explain relationships between variables.
Using data to answer research questions and draw conclusions.
Identifying conclusions supported by research results.
Determining the implications of results for real-world situations.
Using structural comparisons to make predictions about chemical properties in an unfamiliar
scenario.

For example, questions from the Psychological, Social, and Biological Foundations of Behavior section
may ask you to demonstrate your use of data-based and statistical reasoning by:
▪
▪
▪
▪

▪
▪
▪
▪
▪

Identifying the correlation between a demographic variable, such as race/ethnicity, gender, or
age, and life expectancy or another health outcome.
Identifying the relationship between demographic variables and health variables reported in a
table or figure.
Explaining why income data are usually reported using the median rather than the mean.
Using reasoning to identify or evaluate what inference is supported by a table of correlations
between different socioeconomic variables and level of participation in different physical
activities.
Using reasoning about the type of comparisons made in an experimental study of cognitive
dissonance and evaluating what the findings imply for attitude and behavior change.
Drawing conclusions about the type of memory affected by an experimental manipulation when
you are shown a graph of findings from a memory experiment.
Distinguishing the kinds of claims that can be made when using longitudinal data, cross-sectional
data, or experimental data in studies of social interaction.
Identifying which conclusion about mathematical understanding in young children is supported
by time data reported in a developmental study.
Evaluating data collected from different types of research studies, such as comparing results
from a qualitative study of mechanisms for coping with stress with results from a quantitative
study of social support networks.

© 2020 Association of American Medical Colleges

13

▪

Using data, such as interviews with cancer patients or a national survey of health behaviors, to
determine a practical application based on a study’s results.

The three questions that follow illustrate Skill 4 questions from, respectively, the Psychological, Social,
and Biological Foundations of Behavior section; the Biological and Biochemical Foundations of Living
Systems section; and the Chemical and Physical Foundations of Biological Systems section of the MCAT
exam.
Skill 4 Example From the Psychological, Social, and Biological Foundations of Behavior Section
Which correlation supports the bystander effect?
A. The number of bystanders is positively correlated with the time it takes for someone to offer
help in the case of an emergency.
B. The number of bystanders is negatively correlated with the time it takes for someone to offer
help in the case of an emergency.
C. The number of bystanders is positively correlated with whether people judge a situation to be
an emergency.
D. The number of bystanders is negatively correlated with whether people judge a situation to be
an emergency.
The correct answer is A. This Skill 4 question assesses knowledge of Content Category 7B, Social
processes that influence human behavior. It is a Skill 4 question because it requires you to engage in
statistical reasoning. This question requires you to understand the distinction between negative and
positive correlations and make a prediction about data based on your knowledge of the bystander
effect.

© 2020 Association of American Medical Colleges

14

Skill 4 Example From the Biological and Biochemical Foundations of Living Systems Section
In the figure, the three curves represent hemoglobin oxygen binding at three different pH values, pH
7.2, pH 7.4, and pH 7.6.

What conclusion can be drawn from these data about the oxygen binding of hemoglobin at different pH
values?
A.
B.
C.
D.

Low pH favors the high-affinity oxygen-binding state.
Low pH favors the low-affinity oxygen-binding state.
Oxygen affinity is independent of pH.
Oxygen binding is noncooperative at low pH.

The correct answer is B. This Skill 4 question draws on knowledge from Content Category 1A, Structure
and function of proteins and their constituent amino acids. This is a Skill 4 question because it asks you
to use data to explain a property of hemoglobin. You must evaluate the hemoglobin oxygen-binding
data for each pH value and compare them to determine the relationship between pH and hemoglobin
oxygen affinity in order to conclude that low pH favors the low-affinity oxygen-binding state.

© 2020 Association of American Medical Colleges

15

Skill 4 Example From the Chemical and Physical Foundations of Biological Systems Section
Four different solutions of a single amino acid were titrated, and the pK values of the solute were
determined.
Solution

pK1

pK2

pK3

1

2.10

3.86

9.82

2

2.10

4.07

9.47

3

2.32

9.76

Not Applicable

4

2.18

9.04

12.48

Which solution contains an amino acid that would be most likely to stabilize an anionic substrate in an
enzyme pocket at physiological pH?
A. Solution 1
B. Solution 2
C. Solution 3
D. Solution 4
The correct answer is D. This Skill 4 question includes a table and assesses knowledge of Content
Category 5D, Structure, function, and reactivity of biologically relevant molecules. Here you see that four
different solutions of a single amino acid were titrated, and the pK values were determined. These
values are found in the table. This is a Skill 4 question because you must recognize a data pattern in the
table, make comparisons, and use those comparisons to make a prediction. Using knowledge of amino
acids and peptide bonds and the patterns you see in the data, you can determine that the N- and Cterminus pK values, roughly 2 and 9 for all solutions, can be ignored since these groups will be involved
in peptide bond formation. With further analyses, you can determine that only Solution 4 will be
cationic at physiological pH.

© 2020 Association of American Medical Colleges

16

Biological and Biochemical Foundations of Living Systems
What Will the Biological and Biochemical Foundations of Living Systems Section Test?
The Biological and Biochemical Foundations of Living Systems section asks you to solve problems by
combining your knowledge of biological and biochemical concepts with your scientific inquiry and
reasoning skills. This section tests processes that are unique to living organisms, such as growing and
reproducing, maintaining a constant internal environment, acquiring materials and energy, sensing and
responding to environmental changes, and adapting. It also tests how cells and organ systems within an
organism act independently and in concert to accomplish these processes, and it asks you to reason
about these processes at various levels of biological organization within a living system.
This section is designed to:
▪
▪
▪
▪
▪

Test introductory-level biology, organic chemistry, and inorganic chemistry concepts.
Test biochemistry concepts at the level taught in many colleges and universities in first-semester
biochemistry courses.
Test cellular and molecular biology topics at the level taught in many colleges and universities in
introductory biology sequences and first-semester biochemistry courses.
Test basic research methods and statistics concepts described by many baccalaureate faculty as
important to success in introductory science courses.
Require you to demonstrate your scientific inquiry and reasoning, research methods, and
statistics skills as applied to the natural sciences.

Test Section

Number of Questions

Time

Biological and Biochemical
Foundations of Living Systems

59

95 minutes

(note that questions are a
combination of passage-based
and discrete questions)

© 2020 Association of American Medical Colleges

17

Scientific Inquiry and Reasoning Skills
As a reminder, the scientific inquiry and reasoning skills you will be asked to demonstrate on this section
of the exam are:
Knowledge of Scientific Concepts and Principles
▪
▪

Demonstrating understanding of scientific concepts and principles.
Identifying the relationships between closely related concepts.

Scientific Reasoning and Problem-Solving
▪
▪

Reasoning about scientific principles, theories, and models.
Analyzing and evaluating scientific explanations and predictions.

Reasoning About the Design and Execution of Research
▪
▪

Demonstrating understanding of important components of scientific research.
Reasoning about ethical issues in research.

Data-Based and Statistical Reasoning
▪
▪

Interpreting patterns in data presented in tables, figures, and graphs.
Reasoning about data and drawing conclusions from them.

© 2020 Association of American Medical Colleges

18

General Mathematical Concepts and Techniques
It’s important for you to know that questions on the natural, behavioral, and social sciences sections will ask
you to use certain mathematical concepts and techniques. As the descriptions of the scientific inquiry and
reasoning skills suggest, some questions will ask you to analyze and manipulate scientific data to show you
can:
▪
▪
▪

▪
▪
▪

▪

Recognize and interpret linear, semilog, and log-log scales and calculate slopes from data found in
figures, graphs, and tables.
Demonstrate a general understanding of significant digits and the use of reasonable numerical
estimates in performing measurements and calculations.
Use metric units, including converting units within the metric system and between metric and English
units (conversion factors will be provided when needed), and dimensional analysis (using units to
balance equations).
Perform arithmetic calculations involving the following: probability, proportion, ratio, percentage, and
square-root estimations.
Demonstrate a general understanding (Algebra II-level) of exponentials and logarithms (natural and
base 10), scientific notation, and solving simultaneous equations.
Demonstrate a general understanding of the following trigonometric concepts: definitions of basic
(sine, cosine, tangent) and inverse (sin‒1, cos‒1, tan‒1) functions; sin and cos values of 0°, 90°, and
180°; relationships between the lengths of sides of right triangles containing angles of 30°, 45°, and
60°.
Demonstrate a general understanding of vector addition and subtraction and the right-hand rule
(knowledge of dot and cross products is not required).

Note also that an understanding of calculus is not required, and a periodic table will be provided during the
exam.

© 2020 Association of American Medical Colleges

19

Resource
You will have access to the periodic table shown while answering questions in this section of the exam.

© 2020 Association of American Medical Colleges

20

Biological and Biochemical Foundations of Living Systems Distribution of Questions by
Discipline, Foundational Concept, and Scientific Inquiry and Reasoning Skill
You may wonder how much biochemistry you’ll see on this section of the MCAT exam, how many
questions you’ll get about a particular foundational concept, or how the scientific inquiry and reasoning
skills will be distributed on your exam. The questions you see are likely to be distributed in the ways
described below. These are the approximate percentages of questions you’ll see for each discipline,
foundational concept, and scientific inquiry and reasoning skill. (These percentages have been
approximated to the nearest 5% and will vary from one test to another for a variety of reasons,
including, but not limited to, controlling for question difficulty, using groups of questions that depend on
a single passage, and using unscored field-test questions on each test form.)
Discipline:
▪
▪
▪
▪

First-semester biochemistry, 25%
Introductory biology, 65%
General chemistry, 5%
Organic chemistry, 5%

Foundational Concept:
▪
▪
▪

Foundational Concept 1, 55%
Foundational Concept 2, 20%
Foundational Concept 3, 25%

Scientific Inquiry and Reasoning Skill:
▪
▪
▪
▪

Skill 1, 35%
Skill 2, 45%
Skill 3, 10%
Skill 4, 10%

© 2020 Association of American Medical Colleges

21

Biological and Biochemical Foundations of Living Systems Framework of Foundational
Concepts and Content Categories
Foundational Concept 1: Biomolecules have unique properties that determine how they contribute to
the structure and function of cells and how they participate in the processes necessary to maintain life.
The content categories for this foundational concept include:
1A. Structure and function of proteins and their constituent amino acids.
1B. Transmission of genetic information from the gene to the protein.
1C. Transmission of heritable information from generation to generation and the processes that
increase genetic diversity.
1D. Principles of bioenergetics and fuel molecule metabolism.
Foundational Concept 2: Highly organized assemblies of molecules, cells, and organs interact to carry
out the functions of living organisms.
The content categories for this foundational concept include:
2A. Assemblies of molecules, cells, and groups of cells within single cellular and multicellular organisms.
2B. The structure, growth, physiology, and genetics of prokaryotes and viruses.
2C. Processes of cell division, differentiation, and specialization.
Foundational Concept 3: Complex systems of tissues and organs sense the internal and external
environments of multicellular organisms, and through integrated functioning, maintain a stable internal
environment within an ever-changing external environment.
The content categories for this foundational concept include:
3A. Structure and functions of the nervous and endocrine systems and ways these systems coordinate
the organ systems.
3B. Structure and integrative functions of the main organ systems.

© 2020 Association of American Medical Colleges

22

How Foundational Concepts and Content Categories Fit Together
The MCAT exam asks you to solve problems by combining your knowledge of concepts with your
scientific inquiry and reasoning skills. The figure below illustrates how foundational concepts, content
categories, and scientific inquiry and reasoning skills intersect when test questions are written.

Foundational Concept 1

Content
Category 1A

Foundational Concept 2

Content
Category 1B

Content
Category 1C

Content
Category 2A

Content
Category 2B

Content
Category 2C

Skill
Skill 1

▪

Skill 2
Skill 3
Skill 4

▪

Each cell represents the point at which foundational
concepts, content categories, and scientific inquiry and
reasoning skills cross.
Test questions are written at the intersections of the
knowledge and skills.

© 2020 Association of American Medical Colleges

23

Understanding the Foundational Concepts and Content Categories in the Biological and
Biochemical Foundations of Living Systems Section
The following are detailed explanations of each foundational concept and related content categories
tested in the Biological and Biochemical Foundations of Living Systems section. To help you prepare for
the MCAT exam, we provide content lists that describe specific topics and subtopics that define each
content category for this section. The same content lists are provided to the writers who develop the
content of the exam. Here is an excerpt from the content list.
EXCERPT FROM BIOLOGICAL AND BIOCHEMICAL FOUNDATONS OF LIVING SYSTEMS OUTLINE

Metabolism of Fatty Acids and Proteins (BIO, BC)
▪
▪
▪

▪
▪
▪
▪

Topic

Description of fatty acids (BC)
Subtopic
Digestion, mobilization, and transport of fats
Oxidation of fatty acids
o Saturated fats
o Unsaturated fats
Ketone bodies (BC)
Anabolism of fats (BIO)
Nontemplate synthesis: biosynthesis of lipids and polysaccharides (BIO)
Metabolism of proteins (BIO)

The abbreviations in parentheses indicate the courses in which undergraduate students at many
colleges and universities learn about the topics and associated subtopics. The course abbreviations are:
▪
▪
▪
▪

BC: first-semester biochemistry
BIO: two-semester sequence of introductory biology
GC: two-semester sequence of general chemistry
OC: two-semester sequence of organic chemistry

In preparing for the MCAT exam, you will be responsible for learning the topics and associated subtopics
at the levels taught at many colleges and universities in the courses listed in parentheses. A small
number of subtopics have course abbreviations indicated in parentheses. In those cases, you are
responsible only for learning the subtopics as they are taught in the course(s) indicated.
Using the excerpt above as an example:
▪

You are responsible for learning about the topic Metabolism of Fatty Acids and Proteins at the
level taught in a typical two-semester introductory biology sequence and in a typical firstsemester biochemistry course.

© 2020 Association of American Medical Colleges

24

▪

▪

You are responsible for learning about the subtopics Anabolism of fats, Nontemplate synthesis:
biosynthesis of lipids and polysaccharides, and Metabolism of proteins only at the levels taught
in a typical two-semester sequence of introductory biology.
You are responsible for learning about the subtopics Description of fatty acids and Ketone
bodies only at the levels taught in a typical first-semester biochemistry course.

Remember that course content at your school may differ from course content at other colleges and
universities. The topics and subtopics described in this and the next two chapters may be covered in
courses with titles that are different from those listed here. Your prehealth advisor and faculty are
important resources for your questions about course content.

Please Note
Topics that appear on multiple content lists will be treated differently. Questions will focus on the
topics as they are described in the narrative for the content category.

© 2020 Association of American Medical Colleges

25

Biological and Biochemical Foundations of Living Systems
Foundational Concept 1
Biomolecules have unique properties that determine how they contribute to the structure and function of
cells and how they participate in the processes necessary to maintain life.
The unique chemical and structural properties of biomolecules determine the roles they play in cells. The
proper functioning of a living system depends on the many components acting harmoniously in response to a
constantly changing environment. Biomolecules are constantly formed or degraded in response to the
perceived needs of the organism.
Content Categories
▪

▪
▪
▪

Category 1A focuses on the structural and functional complexity of proteins, which is derived from
their component amino acids, the sequence in which the amino acids are covalently bonded, and the
three-dimensional structures the proteins adopt in an aqueous environment.
Category 1B focuses on the molecular mechanisms responsible for the transfer of sequence-specific
biological information between biopolymers that ultimately result in the synthesis of proteins.
Category 1C focuses on the mechanisms that function to transmit the heritable information stored in
DNA from generation to generation.
Category 1D focuses on the biomolecules and regulated pathways involved in harvesting chemical
energy stored in fuel molecules, which serves as the driving force for all the processes that take place
within a living system.

With these building blocks, medical students will be able to learn how the major biochemical, genetic, and
molecular functions of the cell support health and lead disease.
1A: Structure and function of proteins and their
constituent amino acids

Amino Acids (BC, OC)

Macromolecules formed from amino acids adopt welldefined, three-dimensional structures with chemical
properties that are responsible for their participation in
virtually every process occurring within and between
cells. The three-dimensional structure of proteins is a
direct consequence of the nature of the covalently
bonded sequence of amino acids, their chemical and
physical properties, and the way the whole assembly
interacts with water.

© 2020 Association of American Medical Colleges

26

▪ Description
o Absolute configuration at the α position
o Amino acids as dipolar ions
o Classifications
▪ Acidic or basic
▪ Hydrophobic or hydrophilic
▪ Reactions
o Sulfur linkage for cysteine and cystine
o Peptide linkage: polypeptides and proteins
o Hydrolysis

Enzymes are proteins that interact in highly regio- and
stereo-specific ways with dissolved solutes. They either
facilitate the chemical transformation of these solutes
or allow for their transport innocuously. Dissolved
solutes compete for protein-binding sites, and protein
conformational dynamics give rise to mechanisms
capable of controlling enzymatic activity.
The infinite variability of potential amino acid
sequences allows for adaptable responses to
pathogenic organisms and materials. The rigidity of
some amino acid sequences makes them suitable for
structural roles in complex living systems.
Content in this category covers a range of protein
behaviors that originate from the unique chemistry of
amino acids themselves. Amino acid classifications and
protein structural elements are covered. Special
emphasis is placed on enzyme catalysis, including
mechanistic considerations, kinetics, models of
enzyme-substrate interaction, and regulation.

Protein Structure (BIO, BC, OC)
▪ Structure
o 1° structure of proteins
o 2° structure of proteins
o 3° structure of proteins; role of proline, cystine,
hydrophobic bonding
o 4° structure of proteins (BIO, BC)
▪ Conformational stability
o Denaturing and folding
o Hydrophobic interactions
o Solvation layer (entropy) (BC)
▪ Separation techniques
o Isoelectric point
o Electrophoresis
Nonenzymatic Protein Function (BIO, BC)
▪ Binding (BC)
▪ Immune system
▪ Motors
Enzyme Structure and Function (BIO, BC)
▪ Function of enzymes in catalyzing biological
reactions
▪ Enzyme classification by reaction type
▪ Reduction of activation energy
▪ Substrates and enzyme specificity
▪ Active Site Model
▪ Induced-Fit Model
▪ Mechanism of catalysis
o Cofactors
o Coenzymes
o Water-soluble vitamins
▪ Effects of local conditions on enzyme activity
Control of Enzyme Activity (BIO, BC)
▪ Kinetics
o General (catalysis)
o Michaelis-Menten
o Cooperativity

© 2020 Association of American Medical Colleges

27

▪ Feedback regulation
▪ Inhibition ― types
o Competitive
o Noncompetitive
o Mixed (BC)
o Uncompetitive (BC)
▪ Regulatory enzymes
o Allosteric enzymes
o Covalently modified enzymes
o Zymogen
1B: Transmission of genetic information from the gene
to the protein
Biomolecules and biomolecular assemblies interact in
specific, highly regulated ways to transfer sequence
information between biopolymers in living organisms.
By storing and transferring biological information, DNA
and RNA enable living organisms to reproduce their
complex components from one generation to the next.
The nucleotide monomers of these biopolymers, being
joined by phosphodiester linkages, form a
polynucleotide molecule with a “backbone” composed
of repeating sugar-phosphate units and “appendages”
of nitrogenous bases. The unique sequence of bases in
each gene provides specific information to the cell.
DNA molecules are composed of two polynucleotides
that spiral around an imaginary axis, forming a double
helix. The two polynucleotides are held together by
hydrogen bonds between the paired bases and van der
Waals interactions between the stacked bases. The
pairing between the bases of two polynucleotides is
very specific, and its complementarity allows for a
precise replication of the DNA molecule.
The DNA inherited by an organism leads to specific
traits by dictating the synthesis of the biomolecules
(RNA molecules and proteins) involved in protein
synthesis. While every cell in a multicellular organism
inherits the same DNA, its expression is precisely

© 2020 Association of American Medical Colleges

28

Nucleic Acid Structure and Function (BIO, BC)
▪ Description
▪ Nucleotides and nucleosides
o Sugar phosphate backbone
o Pyrimidine, purine residues
▪ Deoxyribonucleic acid (DNA): double helix,
Watson-Crick model of DNA structure
▪ Base pairing specificity: A with T, G with C
▪ Function in transmission of genetic information
(BIO)
▪ DNA denaturation, reannealing, hybridization
DNA Replication (BIO)
▪ Mechanism of replication: separation of strands,
specific coupling of free nucleic acids
▪ Semiconservative nature of replication
▪ Specific enzymes involved in replication
▪ Origins of replication, multiple origins in
eukaryotes
▪ Replicating the ends of DNA molecules
Repair of DNA (BIO)
▪ Repair during replication
▪ Repair of mutations
Genetic Code (BIO)
▪ Central Dogma: DNA → RNA → protein
▪ The triplet code

regulated such that different genes are expressed by
cells at different stages of development, by cells in
different tissues, and by cells exposed to different
stimuli.
The topics included in this category concern not only
the molecular mechanisms of the transmission of
genetic information from the gene to the protein
(transcription and translation), but also the
biosynthesis of the important molecules and molecular
assemblies involved in these mechanisms. The control
of gene expression in prokaryotes and eukaryotes is
also included.
Broadly speaking, the field of biotechnology uses
biological systems, living organisms, or derivatives
thereof to make or modify products or processes for
specific use. The biotechnological techniques
emphasized in this category, however, are those that
take advantage of the complementary structure of
double-stranded DNA molecules to synthesize,
sequence, and amplify them and to analyze and
identify unknown polynucleotide sequences. Included
within this treatment of biotechnology are those
practical applications that directly impact humans, such
as medical applications, human gene therapy, and
pharmaceuticals.
Content in this category covers the biopolymers,
including ribonucleic acid (RNA), deoxyribonucleic acid
(DNA), proteins, and the biochemical processes
involved in carrying out the transfer of biological
information from DNA.

▪ Codon-anticodon relationship
▪ Degenerate code, wobble pairing
▪ Missense, nonsense codons
▪ Initiation, termination codons
▪ Messenger RNA (mRNA)
Transcription (BIO)
▪ Transfer RNA (tRNA); ribosomal RNA (rRNA)
▪ Mechanism of transcription
▪ mRNA processing in eukaryotes, introns, exons
▪ Ribozymes, spliceosomes, small nuclear
ribonucleoproteins (snRNPs), small nuclear RNAs
(snRNAs)
▪ Functional and evolutionary importance of
introns
Translation (BIO)
▪ Roles of mRNA, tRNA, rRNA
▪ Role and structure of ribosomes
▪ Initiation, termination co-factors
▪ Post-translational modification of proteins
Eukaryotic Chromosome Organization (BIO)
▪ Chromosomal proteins
▪ Single copy vs. repetitive DNA
▪ Supercoiling
▪ Heterochromatin vs. euchromatin
▪ Telomeres, centromeres
Control of Gene Expression in Prokaryotes (BIO)
▪ Operon Concept, Jacob-Monod Model
▪ Gene repression in bacteria
▪ Positive control in bacteria
Control of Gene Expression in Eukaryotes (BIO)
▪ Transcriptional regulation
▪ DNA binding proteins, transcription factors
▪ Gene amplification and duplication
▪ Post-transcriptional control, basic concept of
splicing (introns, exons)

© 2020 Association of American Medical Colleges

29

▪ Cancer as a failure of normal cellular controls,
oncogenes, tumor suppressor genes
▪ Regulation of chromatin structure
▪ DNA methylation
▪ Role of noncoding RNAs
Recombinant DNA and Biotechnology (BIO)
▪ Gene cloning
▪ Restriction enzymes
▪ DNA libraries
▪ Generation of cDNA
▪ Hybridization
▪ Expressing cloned genes
▪ Polymerase chain reaction
▪ Gel electrophoresis and Southern blotting
▪ DNA sequencing
▪ Analyzing gene expression
▪ Determining gene function
▪ Stem cells
▪ Practical applications of DNA technology: medical
applications, human gene therapy,
pharmaceuticals, forensic evidence,
environmental cleanup, agriculture
▪ Safety and ethics of DNA technology
1C: Transmission of heritable information from
generation to generation and the processes that
increase genetic diversity

Evidence That DNA Is Genetic Material (BIO)
Mendelian Concepts (BIO)

The information necessary to direct life functions is
contained within discrete nucleotide sequences
transmitted from generation to generation by
mechanisms that, by nature of their various processes,
provide the raw materials for evolution by increasing
genetic diversity. Specific sequences of
deoxyribonucleic acids store and transfer the heritable
information necessary for the continuation of life from
one generation to the next. These sequences, called
genes ― being part of longer DNA molecules ― are

© 2020 Association of American Medical Colleges

30

▪ Phenotype and genotype
▪ Gene
▪ Locus
▪ Allele: single and multiple
▪ Homozygosity and heterozygosity
▪ Wild-type
▪ Recessiveness
▪ Complete dominance
▪ Co-dominance
▪ Incomplete dominance, leakage, penetrance,
expressivity
▪ Hybridization: viability

▪ Gene pool

organized, along with various proteins, into
biomolecular assemblies called chromosomes.
Chromosomes pass from parents to offspring in
sexually reproducing organisms. The processes of
meiosis and fertilization maintain a species’
chromosome count during the sexual life cycle.
Because parents pass on discrete heritable units that
retain their separate identities in offspring, the laws of
probability can be used to predict the outcome of
some, but not all, genetic crosses.
The behavior of chromosomes during meiosis and
fertilization is responsible for most of the genetic
variation that arises each generation. Mechanisms that
contribute to this genetic variation include
independent assortment of chromosomes, crossing
over, and random fertilization. Other mechanisms, such
as mutation, random genetic drift, bottlenecks, and
immigration, exist with the potential to affect the
genetic diversity of individuals and populations.
Collectively, the genetic diversity that results from
these processes provides the raw material for evolution
by natural selection.
The content in this category covers the mechanisms by
which heritable information is transmitted from
generation to generation and the evolutionary
processes that generate and act on genetic variation.

Meiosis and Other Factors Affecting Genetic
Variability (BIO)
▪ Significance of meiosis
▪ Important differences between meiosis and
mitosis
▪ Segregation of genes
o Independent assortment
o Linkage
o Recombination
▪ Single crossovers
▪ Double crossovers
▪ Synaptonemal complex
▪ Tetrad
o Sex-linked characteristics
o Very few genes on Y chromosome
o Sex determination
o Cytoplasmic/extranuclear inheritance
▪ Mutation
o General concept of mutation — error in DNA
sequence
o Types of mutations: random, translation error,
transcription error, base substitution, inversion,
addition, deletion, translocation, mispairing
o Advantageous vs. deleterious mutation
o Inborn errors of metabolism
o Relationship of mutagens to carcinogens
▪ Genetic drift
▪ Synapsis or crossing-over mechanism for
increasing genetic diversity
Analytic Methods (BIO)
▪ Hardy-Weinberg Principle
▪ Testcross (Backcross; concepts of parental, F1,
and F2 generations)
▪ Gene mapping: crossover frequencies
▪ Biometry: statistical methods

© 2020 Association of American Medical Colleges

31

Evolution (BIO)
▪ Natural selection
o Fitness concept
o Selection by differential reproduction
o Concepts of natural and group selection
o Evolutionary success as increase in percentage
representation in the gene pool of the next
generation
▪ Speciation
o Polymorphism
o Adaptation and specialization
o Inbreeding
o Outbreeding
o Bottlenecks
▪ Evolutionary time as measured by gradual
random changes in genome
1D: Principles of bioenergetics and fuel molecule
metabolism

Principles of Bioenergetics (BC, GC)

Living things harness energy from fuel molecules in a
controlled manner that sustains all the processes
responsible for maintaining life. Cell maintenance and
growth is energetically costly. Cells harness the energy
stored in fuel molecules, such as carbohydrates and
fatty acids, and convert it into smaller units of chemical
potential known as adenosine triphosphate (ATP).
The hydrolysis of ATP provides a ready source of energy
for cells that can be coupled to other chemical
processes that make them thermodynamically
favorable. Fuel molecule mobilization, transport, and
storage are regulated according to the needs of the
organism.
The content in this category covers the principles of
bioenergetics and fuel molecule catabolism. Details of
oxidative phosphorylation including the role of
chemiosmotic coupling and biological electron transfer
reactions are covered, as are the general features of
fatty acid and glucose metabolism. Additionally,
© 2020 Association of American Medical Colleges

32

▪ Bioenergetics/thermodynamics
▪ Free energy/Keq
o Equilibrium constant
o Relationship of the equilibrium constant and
ΔG°
▪ Concentration
o Le Châtelier’s Principle
▪ Endothermic and exothermic reactions
▪ Free energy: G
▪ Spontaneous reactions and ΔG°
▪ Phosphoryl group transfers and ATP
o ATP hydrolysis ΔG << 0
o ATP group transfers
▪ Biological oxidation-reduction
o Half-reactions
o Soluble electron carriers
o Flavoproteins

regulation of these metabolic pathways, fuel molecule
mobilization, transport, and storage are covered.

Carbohydrates (BC, OC)
▪ Description
o Nomenclature and classification, common
names
o Absolute configuration
o Cyclic structure and conformations of hexoses
o Epimers and anomers
▪ Hydrolysis of the glycoside linkage
▪ Monosaccharides
▪ Disaccharides
▪ Polysaccharides
Glycolysis, Gluconeogenesis, and the Pentose
Phosphate Pathway (BIO, BC)
▪ Glycolysis (aerobic), substrates and products
o Feeder pathways: glycogen, starch metabolism
▪ Fermentation (anaerobic glycolysis)
▪ Gluconeogenesis (BC)
▪ Pentose phosphate pathway (BC)
▪ Net molecular and energetic results of respiration
processes
Principles of Metabolic Regulation (BC)
▪ Regulation of metabolic pathways (BIO, BC)
o Maintenance of a dynamic steady state
▪ Regulation of glycolysis and gluconeogenesis
▪ Metabolism of glycogen
▪ Regulation of glycogen synthesis and breakdown
o Allosteric and hormonal control
▪ Analysis of metabolic control
Citric Acid Cycle (BIO, BC)
▪ Acetyl-CoA production (BC)
▪ Reactions of the cycle, substrates and products
▪ Regulation of the cycle
▪ Net molecular and energetic results of respiration
processes

© 2020 Association of American Medical Colleges

33

Metabolism of Fatty Acids and Proteins (BIO, BC)
▪ Description of fatty acids (BC)
▪ Digestion, mobilization, and transport of fats
▪ Oxidation of fatty acids
o Saturated fats
o Unsaturated fats
▪ Ketone bodies (BC)
▪ Anabolism of fats (BIO)
▪ Nontemplate synthesis: biosynthesis of lipids and
polysaccharides (BIO)
▪ Metabolism of proteins (BIO)
Oxidative Phosphorylation (BIO, BC)
▪ Electron transport chain and oxidative
phosphorylation, substrates and products,
general features of the pathway
▪ Electron transfer in mitochondria
o NADH, NADPH
o Flavoproteins
o Cytochromes
▪ ATP synthase, chemiosmotic coupling
o Proton motive force
▪ Net molecular and energetic results of respiration
processes
▪ Regulation of oxidative phosphorylation
▪ Mitochondria, apoptosis, oxidative stress (BC)
Hormonal Regulation and Integration of
Metabolism (BC)
▪ Higher-level integration of hormone structure
and function
▪ Tissue-specific metabolism
▪ Hormonal regulation of fuel metabolism
▪ Obesity and regulation of body mass

© 2020 Association of American Medical Colleges

34

Biological and Biochemical Foundations of Living Systems
Foundational Concept 2
Highly organized assemblies of molecules, cells, and organs interact to carry out the functions of living
organisms.
Cells are the basic unit of structure in all living things. Mechanisms of cell division provide not only for the
growth and maintenance of organisms, but also for the continuation of the species through asexual and
sexual reproduction. The unique microenvironment to which a cell is exposed during development and
division determines the fate of the cell by impacting gene expression and ultimately the cell’s collection and
distribution of macromolecules and its arrangement of subcellular organelles.
In multicellular organisms, the processes necessary to maintain life are executed by groups of cells organized
into specialized structures with specialized functions ― both of which result from the unique properties of
the cells’ component molecules.
Content Categories
▪
▪
▪

Category 2A focuses on the assemblies of molecules, cells, and groups of cells within single cellular
and multicellular organisms that function to execute the processes necessary to maintain life.
Category 2B focuses on the structure, growth, physiology, and genetics of prokaryotes and the
structure and life cycles of viruses.
Category 2C focuses on the processes of cell and nuclear division and the mechanisms governing cell
differentiation and specialization.

With these building blocks, medical students will be able to learn how cells grow and integrate to form tissues
and organs that carry out essential biochemical and physiological functions.
2A: Assemblies of molecules, cells, and groups of cells
within single cellular and multicellular organisms
The processes necessary to maintain life are executed
by assemblies of molecules, cells, and groups of cells,
all of which are organized into highly specific
structures as determined by the unique properties of
their component molecules. The processes necessary
to maintain life require that cells create and maintain
internal environments within the cytoplasm and

© 2020 Association of American Medical Colleges

35

Plasma Membrane (BIO, BC)
▪ General function in cell containment
▪ Composition of membranes
o Lipid components (BIO, BC, OC)
▪ Phospholipids (and phosphatids)
▪ Steroids
▪ Waxes
o Protein components
o Fluid mosaic model
▪ Membrane dynamics

▪ Solute transport across membranes
o Thermodynamic considerations
o Osmosis
Cell membranes separate the internal environment of
▪ Colligative properties; osmotic pressure (GC)
the cell from the external environment. The
o Passive transport
specialized structure of the membrane, as described in
o Active transport
the fluid mosaic model, allows the cell to be
▪ Sodium/potassium pump
selectively permeable and dynamic, with homeostasis
▪ Membrane channels
maintained by the constant movement of molecules
▪ Membrane potential
across the membranes through a combination of
▪ Membrane receptors
active and passive processes driven by several forces,
▪ Exocytosis and endocytosis
including electrochemical gradients.
▪ Intercellular junctions (BIO)
o Gap junctions
Eukaryotic cells also maintain internal membranes
o Tight junctions
that partition the cell into specialized regions. These
o Desmosomes
internal membranes facilitate cellular processes by
minimizing conflicting interactions and increasing
Membrane-Bound Organelles and Defining
surface area where chemical reactions can occur.
Characteristics of Eukaryotic Cells (BIO)
Membrane-bound organelles localize different
processes or enzymatic reactions in time and space.
▪ Defining characteristics of eukaryotic cells:
membrane-bound nucleus, presence of organelles,
Through interactions between proteins bound to the
mitotic division
membranes of adjacent cells or between membrane▪ Nucleus
bound proteins and elements of the extracellular
o Compartmentalization, storage of genetic
matrix, cells of multicellular organisms organize into
information
tissues, organs, and organ systems. Certain
o Nucleolus: location and function
membrane-associated proteins also play key roles in
o Nuclear envelope, nuclear pores
identifying tissues or recent events in the cell’s history
▪ Mitochondria
for purposes of recognition of “self” versus foreign
o Site of ATP production
molecules.
o Inner- and outer-membrane structure (BIO, BC)
o Self-replication
The content in this category covers the composition,
▪
Lysosomes: membrane-bound vesicles containing
structure, and function of cell membranes; the
hydrolytic enzymes
structure and function of the membrane-bound
▪ Endoplasmic reticulum
organelles of eukaryotic cells; and the structure and
o Rough and smooth components
function of the major cytoskeletal elements. It covers
o Rough endoplasmic reticulum site of ribosomes
the energetics of and mechanisms by which
o Double-membrane structure
molecules, or groups of molecules, move across cell
o Role in membrane biosynthesis
membranes. It also covers how cell-cell junctions and
o Role in biosynthesis of secreted proteins
the extracellular matrix interact to form tissues with
▪ Golgi apparatus: general structure and role in
packaging and secretion
within certain organelles that are different from their
external environments.

© 2020 Association of American Medical Colleges

36

specialized functions. Epithelial tissue and connective
tissue are covered in this category.

▪ Peroxisomes: organelles that collect peroxides
Cytoskeleton (BIO)
▪ General function in cell support and movement
▪ Microfilaments: composition and role in cleavage
and contractility
▪ Microtubules: composition and role in support and
transport
▪ Intermediate filaments, role in support
▪ Composition and function of cilia and flagella
▪ Centrioles, microtubule-organizing centers
Tissues Formed From Eukaryotic Cells (BIO)
▪ Epithelial cells
▪ Connective tissue cells

2B: The structure, growth, physiology, and genetics
of prokaryotes and viruses
The highly organized assembly of molecules that is the
cell represents the fundamental unit of structure,
function, and organization in all living organisms. In
the hierarchy of biological organization, the cell is the
simplest collection of matter capable of carrying out
the processes that distinguish living organisms. As
such, cells have the ability to undergo metabolism;
maintain homeostasis, including ionic gradients; grow;
move in response to their local environments;
respond to stimuli; reproduce; and adapt to their
environment in successive generations.
Life at cellular levels arises from structural order and
its dynamic modulation. This happens in response to
signals, thereby reflecting properties that result from
individual and interactive features of molecular
assemblies, their compartmentalization, and their
interaction with environmental signals at many spatial
and temporal scales.
The content in this category covers the classification,
structure, growth, physiology, and genetics of
© 2020 Association of American Medical Colleges

37

Cell Theory (BIO)
▪ History and development
▪ Impact on biology
Classification and Structure of Prokaryotic Cells
(BIO)
▪ Prokaryotic domains
o Archaea
o Bacteria
▪ Major classifications of bacteria by shape
o Bacilli (rod-shaped)
o Spirilli (spiral-shaped)
o Cocci (spherical)
▪ Lack of nuclear membrane and mitotic apparatus
▪ Lack of typical eukaryotic organelles
▪ Presence of cell wall in bacteria
▪ Flagellar propulsion, mechanism
Growth and Physiology of Prokaryotic Cells (BIO)
▪ Reproduction by fission
▪ High degree of genetic adaptability, acquisition of
antibiotic resistance

prokaryotes and the characteristics that distinguish
them from eukaryotes. Viruses are also covered here.

▪ Exponential growth
▪ Existence of anaerobic and aerobic variants
▪ Parasitic and symbiotic
▪ Chemotaxis
Genetics of Prokaryotic Cells (BIO)
▪ Existence of plasmids, extragenomic DNA
▪ Transformation: incorporation into bacterial
genome of DNA fragments from external medium
▪ Conjugation
▪ Transposons (also present in eukaryotic cells)
Virus Structure (BIO)
▪ General structural characteristics (nucleic acid and
protein, enveloped and nonenveloped)
▪ Lack organelles and nucleus
▪ Structural aspects of typical bacteriophage
▪ Genomic content — RNA or DNA
▪ Size relative to bacteria and eukaryotic cells
Viral Life Cycle (BIO)
▪ Self-replicating biological units that must
reproduce within specific host cell
▪ Generalized phage and animal virus life cycles
o Attachment to host, penetration of cell
membrane or cell wall, and entry of viral genetic
material
o Use of host synthetic mechanism to replicate
viral components
o Self-assembly and release of new viral particles
▪ Transduction: transfer of genetic material by
viruses
▪ Retrovirus life cycle: integration into host DNA,
reverse transcriptase, HIV
▪ Prions and viroids: subviral particles

2C: Processes of cell division, differentiation, and
specialization

Mitosis (BIO)
▪ Mitotic process: prophase, metaphase, anaphase,
telophase, interphase

© 2020 Association of American Medical Colleges

38

The ability of organisms to reproduce their own kind is
the characteristic that best distinguishes living things.
In sexually reproducing organisms, the continuity of
life is based on the processes of cell division and
meiosis.
The process of cell division is an integral part of the
cell cycle. The progress of eukaryotic cells through the
cell cycle is regulated by a complex molecular control
system. Malfunctions in this system can result in
unabated cellular division and, ultimately, the
development of cancer.

▪ Mitotic structures
o Centrioles, asters, spindles
o Chromatids, centromeres, kinetochores
o Nuclear membrane breakdown and
reorganization
o Mechanisms of chromosome movement
▪ Phases of cell cycle: G0, G1, S, G2, M
▪ Growth arrest
▪ Control of cell cycle
▪ Loss of cell cycle controls in cancer cells

Biosignaling (BC)
In the embryonic development of multicellular
organisms, a fertilized egg gives rise to cells that
▪ Oncogenes, apoptosis
differentiate into many different types of cells, each
Reproductive System (BIO)
with a different structure, corresponding function, and
location within the organism. During development,
▪ Gametogenesis by meiosis
spatial-temporal gradients in the interactions between ▪ Ovum and sperm
gene expression and various stimuli result in the
o Differences in formation
structural and functional divergence of cells into
o Differences in morphology
specialized structures, organs, and tissues. The
o Relative contribution to next generation
interaction of stimuli and genes is also explained by
▪ Reproductive sequence: fertilization, implantation,
the progression of stem cells to terminal cells.
development, birth
The content in this category covers the cell cycle; the
causes, genetics, and basic properties of cancer; the
processes of meiosis and gametogenesis; and the
mechanisms governing cell specialization and
differentiation.

© 2020 Association of American Medical Colleges

39

Embryogenesis (BIO)
▪ Stages of early development (order and general
features of each)
o Fertilization
o Cleavage
o Blastula formation
o Gastrulation
▪ First cell movements
▪ Formation of primary germ layers (endoderm,
mesoderm, ectoderm)
▪ Neurulation
▪ Major structures arising out of primary germ layers
▪ Neural crest
▪ Environment-gene interaction in development

Mechanisms of Development (BIO)
▪ Cell specialization
o Determination
o Differentiation
o Tissue types
▪ Cell-cell communication in development
▪ Cell migration
▪ Pluripotency: stem cells
▪ Gene regulation in development
▪ Programmed cell death
▪ Existence of regenerative capacity in various
species
▪ Senescence and aging

© 2020 Association of American Medical Colleges

40

Biological and Biochemical Foundations of Living Systems
Foundational Concept 3
Complex systems of tissues and organs sense the internal and external environments of multicellular
organisms and, through integrated functioning, maintain a stable internal environment.
As a result of the integration of a number of highly specialized organ systems, complex living things are able to
maintain homeostasis while adapting to a constantly changing environment and participating in growth and
reproduction. The interactions of these organ systems involve complex regulatory mechanisms that help
maintain a dynamic and healthy equilibrium, regardless of the organ systems’ current state and environment.
Content Categories
▪

▪

Category 3A focuses on the structure and functions of the nervous and endocrine systems and the
ways the systems work together to coordinate the responses of other body systems to both external
and internal stimuli.
Category 3B focuses on the structure and functions of the organ systems ― circulatory, respiratory,
digestive, immune, lymphatic, muscular, skeletal, and reproductive ― and the ways these systems
interact to fulfill their concerted roles in the maintenance and continuance of the living organism.

With these building blocks, medical students will be able to learn how the body responds to internal and
external stimuli to support homeostasis and the ability to reproduce.
3A: Structure and functions of the nervous and
endocrine systems and ways these systems coordinate
the organ systems
The nervous and endocrine systems work together to
detect external and internal signals, transmit and
integrate information, and maintain homeostasis. They
do all this by producing appropriate responses to
internal and external cues and stressors. The
integration of these systems both with one another
and with the other organ systems ultimately results in
the successful and adaptive behaviors that allow for
the propagation of the species.
Animals have evolved a nervous system that senses
and processes internal and external information used
to facilitate and enhance survival, growth, and
reproduction. The nervous system interfaces with
© 2020 Association of American Medical Colleges

41

Nervous System: Structure and Function (BIO)
▪ Major functions
o High-level control and integration of body
systems
o Adaptive capability to external influences
▪ Organization of vertebrate nervous system
▪ Sensor and effector neurons
▪ Sympathetic and parasympathetic nervous
systems: antagonistic control
▪ Reflexes
o Feedback loop, reflex arc
o Role of spinal cord and supraspinal circuits
▪ Integration with endocrine system: feedback
control

sensory and internal body systems to coordinate
physiological and behavioral responses ranging from
simple movements and small metabolic changes to
long-distance migrations and social interactions. The
physiological processes for nerve signal generation and
propagation involve specialized membranes with
associated proteins that respond to ligands and/or
electrical field changes, signaling molecules, and, by
extension, the establishment and replenishment of
ionic electrochemical gradients requiring ATP.
The endocrine system of animals has changed over
time to produce chemical signals that function
internally to regulate stress responses, reproduction,
development, energy metabolism, growth, and various
individual and interactive behaviors. The integrated
contributions of the nervous and endocrine systems to
bodily functions are exemplified by the process
whereby the signaling of neurons regulates hormone
release and by the targeting of membrane or nuclear
receptors on neurons by circulating hormones.
The content in this category covers the structure,
function, and basic aspects of nervous and endocrine
systems and their integration. The structure and
function of nerve cells is also included in this category.

Nerve Cell (BIO)
▪ Cell body: site of nucleus, organelles
▪ Dendrites: branched extensions of cell body
▪ Axon: structure and function
▪ Myelin sheath, Schwann cells, insulation of axon
▪ Nodes of Ranvier: propagation of nerve impulse
along axon
▪ Synapse: site of impulse propagation between
cells
▪ Synaptic activity: transmitter molecules
▪ Resting potential: electrochemical gradient
▪ Action potential
o Threshold, all-or-none
o Sodium-potassium pump
▪ Excitatory and inhibitory nerve fibers: summation,
frequency of firing
▪ Glial cells, neuroglia
Electrochemistry (GC)
▪ Concentration cell: direction of electron flow,
Nernst equation
Biosignaling (BC)
▪ Gated ion channels
o Voltage gated
o Ligand gated
▪ Receptor enzymes
▪ G protein-coupled receptors
Lipids (BC, OC)
▪ Description; structure
o Steroids
o Terpenes and terpenoids
Endocrine System: Hormones and Their Sources
(BIO)
▪ Function of endocrine system: specific chemical
control at cell, tissue, and organ level
▪ Definitions of endocrine gland, hormone

© 2020 Association of American Medical Colleges

42

▪ Major endocrine glands: names, locations,
products
▪ Major types of hormones
▪ Neuroendocrinology ― relation between neurons
and hormonal systems
Endocrine System: Mechanisms of Hormone Action
(BIO)
▪ Cellular mechanisms of hormone action
▪ Transport of hormones: blood supply
▪ Specificity of hormones: target tissue
▪ Integration with nervous system: feedback control
▪ Regulation by second messengers
3B: Structure and integrative functions of the main
organ systems
Animals use a number of highly organized and
integrated organ systems to carry out the necessary
functions associated with maintaining life processes.
Within the body, no organ system is an island.
Interactions and coordination between organ systems
allow organisms to engage in the processes necessary
to sustain life. For example, the organs and structures
of the circulatory system carry out a number of
functions, such as transporting:
▪ Nutrients absorbed in the digestive system.
▪ Gases absorbed from the respiratory system and
muscle tissue.
▪ Hormones secreted from the endocrine system.
▪ Blood cells, produced in bone marrow, to and from
cells in the body to help fight disease.
The content in this category covers the structure and
function of the major organ systems of the body,
including the respiratory, circulatory, lymphatic,
immune, digestive, excretory, reproductive, muscle,
skeletal, and skin systems. Also covered in this category
is the integration of these systems and their control

© 2020 Association of American Medical Colleges

43

Respiratory System (BIO)
▪ General function
o Gas exchange, thermoregulation
o Protection against disease: particulate matter
▪ Structure of lungs and alveoli
▪ Breathing mechanisms
o Diaphragm, rib cage, differential pressure
o Resiliency and surface tension effects
▪ Thermoregulation: nasal and tracheal capillary
beds; evaporation, panting
▪ Particulate filtration: nasal hairs, mucus-cilia
system in lungs
▪ Alveolar gas exchange
o Diffusion, differential partial pressure
o Henry’s Law (GC)
▪ pH control
▪ Regulation by nervous control
o CO2 sensitivity
Circulatory System (BIO)
▪ Functions: circulation of oxygen, nutrients,
hormones, ions and fluids, removal of metabolic
waste
▪ Role in thermoregulation
▪ Four-chambered heart: structure and function

▪ Endothelial cells
▪ Systolic and diastolic pressure
▪ Pulmonary and systemic circulation
▪ Arterial and venous systems (arteries, arterioles,
venules, veins)
o Structural and functional differences
o Pressure and flow characteristics
▪ Capillary beds
o Mechanisms of gas and solute exchange
o Mechanism of heat exchange
o Source of peripheral resistance
▪ Composition of blood
o Plasma, chemicals, blood cells
o Erythrocyte production and destruction; spleen,
bone marrow
o Regulation of plasma volume
▪ Coagulation, clotting mechanisms
▪ Oxygen transport by blood
o Hemoglobin, hematocrit
o Oxygen content
o Oxygen affinity
▪ Carbon dioxide transport and level in blood
▪ Nervous and endocrine control

and coordination by the endocrine and nervous
systems.

Lymphatic System (BIO)
▪ Structure of lymphatic system
▪ Major functions
o Equalization of fluid distribution
o Transport of proteins and large glycerides
o Production of lymphocytes involved in immune
reactions
o Return of materials to the blood
Immune System (BIO)
▪ Innate (nonspecific) vs. adaptive (specific)
immunity
▪ Adaptive immune system cells
o T-lymphocytes
o B-lymphocytes

© 2020 Association of American Medical Colleges

44

▪ Innate immune system cells
o Macrophages
o Phagocytes
▪ Tissues
o Bone marrow
o Spleen
o Thymus
o Lymph nodes
▪ Concept of antigen and antibody
▪ Antigen presentation
▪ Clonal selection
▪ Antigen-antibody recognition
▪ Structure of antibody molecule
▪ Recognition of self vs. nonself, autoimmune
diseases
▪ Major histocompatibility complex
Digestive System (BIO)
▪ Ingestion
o Saliva as lubrication and source of enzymes
o Ingestion; esophagus, transport function
▪ Stomach
o Storage and churning of food
o Low pH, gastric juice, mucal protection against
self-destruction
o Production of digestive enzymes, site of
digestion
o Structure (gross)
▪ Liver
o Structural relationship of liver within
gastrointestinal system
o Production of bile
o Role in blood glucose regulation, detoxification
▪ Bile
o Storage in gall bladder
o Function
▪ Pancreas
o Production of enzymes
o Transport of enzymes to small intestine

© 2020 Association of American Medical Colleges

45

▪ Small intestine
o Absorption of food molecules and water
o Function and structure of villi
o Production of enzymes, site of digestion
o Neutralization of stomach acid
o Structure (anatomic subdivisions)
▪ Large intestine
o Absorption of water
o Bacterial flora
o Structure (gross)
▪ Rectum: storage and elimination of waste, feces
▪ Muscular control
o Peristalsis
▪ Endocrine control
o Hormones
o Target tissues
▪ Nervous control: the enteric nervous system
Excretory System (BIO)
▪ Roles in homeostasis
o Blood pressure
o Osmoregulation
o Acid-base balance
o Removal of soluble nitrogenous waste
▪ Kidney structure
o Cortex
o Medulla
▪ Nephron structure
o Glomerulus
o Bowman’s capsule
o Proximal tubule
o Loop of Henle
o Distal tubule
o Collecting duct
▪ Formation of urine
o Glomerular filtration
o Secretion and reabsorption of solutes
o Concentration of urine
o Counter-current multiplier mechanism
▪ Storage and elimination: ureter, bladder, urethra
© 2020 Association of American Medical Colleges

46

▪ Osmoregulation: capillary reabsorption of H2O,
amino acids, glucose, ions
▪ Muscular control: sphincter muscle
Reproductive System (BIO)
▪ Male and female reproductive structures and their
functions
o Gonads
o Genitalia
o Differences between male and female structures
▪ Hormonal control of reproduction
o Male and female sexual development
o Female reproductive cycle
o Pregnancy, parturition, lactation
o Integration with nervous control
Muscle System (BIO)
▪ Important functions
o Support: mobility
o Peripheral circulatory assistance
o Thermoregulation (shivering reflex)
▪ Structure of three basic muscle types: striated,
smooth, cardiac
▪ Muscle structure and control of contraction
o T-tubule system
o Contractile apparatus
o Sarcoplasmic reticulum
o Fiber type
o Contractile velocity of different muscle types
▪ Regulation of cardiac muscle contraction
▪ Oxygen debt: fatigue
▪ Nervous control
o Motor neurons
o Neuromuscular junction, motor end plates
o Sympathetic and parasympathetic innervation
o Voluntary and involuntary muscles

© 2020 Association of American Medical Colleges

47

Specialized Cell ― Muscle Cell (BIO)
▪ Structural characteristics of striated, smooth, and
cardiac muscle
▪ Abundant mitochondria in red muscle cells: ATP
source
▪ Organization of contractile elements: actin and
myosin filaments, crossbridges, sliding filament
model
▪ Sarcomeres: “I” and “A” bands, “M” and “Z” lines,
“H” zone
▪ Presence of troponin and tropomyosin
▪ Calcium regulation of contraction
Skeletal System (BIO)
▪ Functions
o Structural rigidity and support
o Calcium storage
o Physical protection
▪ Skeletal structure
o Specialization of bone types, structures
o Joint structures
o Endoskeleton vs. exoskeleton
▪ Bone structure
o Calcium-protein matrix
o Cellular composition of bone
▪ Cartilage: structure and function
▪ Ligaments, tendons
▪ Endocrine control
Skin System (BIO)
▪ Structure
o Layer differentiation, cell types
o Relative impermeability to water
▪ Functions in homeostasis and osmoregulation
▪ Functions in thermoregulation
o Hair, erectile musculature
o Fat layer for insulation
o Sweat glands, location in dermis

© 2020 Association of American Medical Colleges

48

o Vasoconstriction and vasodilation in surface
capillaries
▪ Physical protection
o Nails, calluses, hair
o Protection against abrasion, disease organisms
▪ Hormonal control: sweating, vasodilation, and
vasoconstriction

© 2020 Association of American Medical Colleges

49

Chemical and Physical Foundations of Biological Systems
What Will the Chemical and Physical Foundations of Biological Systems Section Test?
The Chemical and Physical Foundations of Biological Systems section asks you to solve problems by
combining your knowledge of chemical and physical foundational concepts with your scientific inquiry
and reasoning skills. This section tests your understanding of the mechanical, physical, and biochemical
functions of human tissues, organs, and organ systems. It also tests your knowledge of the basic
chemical and physical principles that underlie the mechanisms operating in the human body and your
ability to reason about and apply your understanding of these basic chemical and physical principles to
living systems.
This section is designed to:
▪
▪
▪
▪
▪

Test introductory-level biology, organic and inorganic chemistry, and physics concepts.
Test biochemistry concepts at the level taught in many colleges and universities in first-semester
biochemistry courses.
Test molecular biology topics at the level taught in many colleges and universities in
introductory biology sequences and first-semester biochemistry courses.
Test basic research methods and statistics concepts described by many baccalaureate faculty as
important to success in introductory science courses.
Require you to demonstrate your scientific inquiry and reasoning, research methods, and
statistics skills as applied to the natural sciences.

Test Section

Number of Questions

Time

Chemical and Physical
Foundations of Biological
Systems

59

95 minutes

(note that questions are a
combination of passage-based
and discrete questions)

© 2020 Association of American Medical Colleges

50

Scientific Inquiry and Reasoning Skills
As a reminder, the scientific inquiry and reasoning skills you will be asked to demonstrate on this section
of the exam are:
Knowledge of Scientific Concepts and Principles
▪
▪

Demonstrating understanding of scientific concepts and principles.
Identifying the relationships between closely related concepts.

Scientific Reasoning and Problem-Solving
▪
▪

Reasoning about scientific principles, theories, and models.
Analyzing and evaluating scientific explanations and predictions.

Reasoning About the Design and Execution of Research
▪
▪

Demonstrating understanding of important components of scientific research.
Reasoning about ethical issues in research.

Data-Based and Statistical Reasoning
▪
▪

Interpreting patterns in data presented in tables, figures, and graphs.
Reasoning about data and drawing conclusions from them.

© 2020 Association of American Medical Colleges

51

General Mathematical Concepts and Techniques
It’s important for you to know that questions on the natural, behavioral, and social sciences sections will ask you
to use certain mathematical concepts and techniques. As the descriptions of the scientific inquiry and reasoning
skills suggest, some questions will ask you to analyze and manipulate scientific data to show you can:
▪
▪
▪

▪
▪
▪

▪

Recognize and interpret linear, semilog, and log-log scales and calculate slopes from data found in figures,
graphs, and tables.
Demonstrate a general understanding of significant digits and the use of reasonable numerical estimates
in performing measurements and calculations.
Use metric units, including converting units within the metric system and between metric and English
units (conversion factors will be provided when needed), and dimensional analysis (using units to balance
equations).
Perform arithmetic calculations involving the following: probability, proportion, ratio, percentage, and
square-root estimations.
Demonstrate a general understanding (Algebra II-level) of exponentials and logarithms (natural and base
10), scientific notation, and solving simultaneous equations.
Demonstrate a general understanding of the following trigonometric concepts: definitions of basic (sine,
cosine, tangent) and inverse (sin‒1, cos‒1, tan‒1) functions; sin and cos values of 0°, 90°, and 180°;
relationships between the lengths of sides of right triangles containing angles of 30°, 45°, and 60°.
Demonstrate a general understanding of vector addition and subtraction and the right-hand rule
(knowledge of dot and cross products is not required)

Note also that an understanding of calculus is not required, and a periodic table will be provided during the exam.

© 2020 Association of American Medical Colleges

52

Resource
You will have access to the periodic table shown while answering questions in this section of the exam.

© 2020 Association of American Medical Colleges

53

Chemical and Physical Foundations of Biological Systems Distribution of Questions by
Discipline, Foundational Concept, and Scientific Inquiry and Reasoning Skill
You may wonder how much chemistry you’ll see on this section of the MCAT exam, how many questions
you’ll get about a particular foundational concept, or how the scientific inquiry and reasoning skills will
be distributed on your exam. The questions you see are likely to be distributed in the ways described
below. These are the approximate percentages of questions you’ll see on a test for each discipline,
foundational concept, and scientific inquiry and reasoning skill. (These percentages have been
approximated to the nearest 5% and will vary from one test to another for a variety of reasons,
including, but not limited to, controlling for question difficulty, using groups of questions that depend on
a single passage, and using unscored field-test questions on each test form.)
Discipline:
▪
▪
▪
▪
▪

First-semester biochemistry, 25%
Introductory biology, 5%
General chemistry, 30%
Organic chemistry, 15%
Introductory physics, 25%

Foundational Concept:
▪
▪

Foundational Concept 4, 40%
Foundational Concept 5, 60%

Scientific Inquiry and Reasoning Skill:
▪
▪
▪
▪

Skill 1, 35%
Skill 2, 45%
Skill 3, 10%
Skill 4, 10%

Chemical and Physical Foundations of Biological Systems Framework of Foundational
Concepts and Content Categories
Foundational Concept 4: Complex living organisms transport materials, sense their environment,
process signals, and respond to changes using processes understood in terms of physical principles.
The content categories for this foundational concept are:
4A. Translational motion, forces, work, energy, and equilibrium in living systems.
4B. Importance of fluids for the circulation of blood, gas movement, and gas exchange.
4C. Electrochemistry and electrical circuits and their elements.
4D. How light and sound interact with matter.
© 2020 Association of American Medical Colleges

54

4E. Atoms, nuclear decay, electronic structure, and atomic chemical behavior.
Foundational Concept 5: The principles that govern chemical interactions and reactions form the basis
for a broader understanding of the molecular dynamics of living systems.
The content categories for this foundational concept are:
5A. Unique nature of water and its solutions.
5B. Nature of molecules and intermolecular interactions.
5C. Separation and purification methods.
5D. Structure, function, and reactivity of biologically relevant molecules.
5E. Principles of chemical thermodynamics and kinetics.

How Foundational Concepts and Content Categories Fit Together
The MCAT exam asks you to solve problems by combining your knowledge of concepts with your
scientific inquiry and reasoning skills. The figure below illustrates how foundational concepts, content
categories, and scientific inquiry and reasoning skills intersect when test questions are written.

Foundational Concept 1

Skill

Content
Category 1A

Foundational Concept 2

Content
Category 1B

Skill 1

▪

Skill 2
Skill 3
Skill 4

▪

Content
Category 1C

Content
Category 2A

Content
Category 2B

Content
Category 2C

Each cell represents the point at which foundational
concepts, content categories, and scientific inquiry and
reasoning skills cross.
Test questions are written at the intersections of the
knowledge and skills.

© 2020 Association of American Medical Colleges

55

Understanding the Foundational Concepts and Content Categories in the Chemical and
Physical Foundations of Biological Systems Outline
The following are detailed explanations of each foundational concept and related content categories
tested in this section. As with the Biological and Biochemical Foundations of Living Systems section, lists
describing the specific topics and subtopics that define each content category for this section are
provided. The same content list is provided to the writers who develop the content of the exam. Here is
an excerpt from the content list.
EXCERPT FROM THE CHEMICAL AND PHYSICAL FOUNDATIONS OF BIOLOGICAL SYSTEMS OUTLINE
Separations and Purifications (OC, BC)
▪
▪
▪

▪

▪

Topic

Extraction: distribution of solute between two immiscible solvents
Distillation
Chromatography: basic principles involved in separation process
o Column chromatography
▪ Gas-liquid chromatography
▪ High pressure liquid chromatography
o Paper chromatography
o Thin-layer chromatography
Separation and purification of peptides and proteins (BC)
o Electrophoresis
o Quantitative analysis
o Chromatography
▪ Size-exclusion
▪ Ion-exchange
▪ Affinity
Racemic mixtures, separation of enantiomers (OC)

Subtopic

The abbreviations in parentheses indicate the course(s) in which undergraduate students at many
colleges and universities learn about the topics and associated subtopics. The course abbreviations are:
▪
▪
▪
▪
▪

BC: first semester of biochemistry
BIO: two-semester sequence of introductory biology
GC: two-semester sequence of general chemistry
OC: two-semester sequence of organic chemistry
PHY: two-semester sequence of introductory physics

In preparing for the MCAT exam, you will be responsible for learning the topics and associated subtopics
at the levels taught at many colleges and universities in the courses listed in parentheses. A small

© 2020 Association of American Medical Colleges

56

number of subtopics have course abbreviations indicated in parentheses. In those cases, you are
responsible only for learning the subtopics as they are taught in the course(s) indicated.
Using the excerpt above as an example:
▪ You are responsible for learning about the topic Separations and Purifications at the level taught
in a typical two-semester organic chemistry sequence and in a typical first-semester
biochemistry course.
▪ You are responsible for learning about the subtopic Separation and purifications of peptides and
proteins (and sub-subtopics) only at the level taught in a first-semester biochemistry course.
▪ You are responsible for learning about the subtopic Racemic mixtures, separation of
enantiomers only at the level taught in a two-semester organic chemistry course.
Remember that course content at your school may differ from course content at other colleges and
universities. The topics and subtopics described in this chapter may be covered in courses with titles
that are different from those listed here. Your prehealth advisor and faculty are important resources for
your questions about course content.

Please Note
Topics that appear on multiple content lists will be treated differently. Questions will focus on the
topics as they are described in the narrative for the content category.

© 2020 Association of American Medical Colleges

57

Chemical and Physical Foundations of Biological Systems
Foundational Concept 4
Complex living organisms transport materials, sense their environment, process signals, and respond to
changes using processes that can be understood in terms of physical principles.
The processes that take place within organisms follow the laws of physics. They can be quantified with
equations that model the behavior at a fundamental level. For example, the principles of electromagnetic
radiation and its interactions with matter can be exploited to generate structural information about molecules
or to generate images of the human body. So, too, can atomic structure be used to predict the physical and
chemical properties of atoms, including the amount of electromagnetic energy required to cause ionization.
Content Categories
▪
▪
▪

▪

▪

Category 4A focuses on motion and its causes and various forms of energy and their interconversions.
Category 4B focuses on the behavior of fluids, which is relevant to the functioning of the pulmonary
and circulatory systems.
Category 4C emphasizes the nature of electrical currents and voltages, how energy can be converted
into electrical forms that can be used to perform chemical transformations or work, and how electrical
impulses can be transmitted over long distances in the nervous system.
Category 4D focuses on the properties of light and sound, how the interactions of light and sound with
matter can be used by an organism to sense its environment, and how these interactions can also be
used to generate structural information or images.
Category 4E focuses on subatomic particles, the atomic nucleus, nuclear radiation, the structure of the
atom, and how the configuration of any particular atom can be used to predict its physical and
chemical properties.

With these building blocks, medical students will be able to use core principles of physics to learn about the
physiological functions of the respiratory, cardiovascular, and neurological systems in health and disease.
4A: Translational motion, forces, work, energy, and
equilibrium in living systems

Translational Motion (PHY)

▪ Units and dimensions
The motion of any object can be described in terms of
▪ Vectors, components
displacement, velocity, and acceleration. Objects
▪ Vector addition
accelerate when subjected to external forces and are at ▪ Speed, velocity (average and instantaneous)
equilibrium when the net force and the net torque
▪ Acceleration
acting on them are zero. Many aspects of motion can
Force (PHY)
be calculated with the knowledge that energy is
conserved, even though it may be converted into
▪ Newton’s First Law, inertia
different forms. In a living system, the energy for
▪ Newton’s Second Law (F = ma)
© 2020 Association of American Medical Colleges

58

motion comes from the metabolism of fuel molecules,
but the energetic requirements remain subject to the
same physical principles.

▪ Newton’s Third Law, forces equal and opposite
▪ Friction, static and kinetic
▪ Center of mass

The content in this category covers several physics
topics relevant to living systems including translational
motion, forces, work, energy, and equilibrium.

Equilibrium (PHY)
▪ Vector analysis of forces acting on a point object
▪ Torques, lever arms
Work (PHY)
▪ Work done by a constant force: W = Fd cosθ
▪ Mechanical advantage
▪ Work Kinetic Energy Theorem
▪ Conservative forces
Energy of Point Object Systems (PHY)
▪ Kinetic Energy: KE = ½mv2; units
▪ Potential Energy
o PE = mgh (gravitational, local)
o PE = ½kx2 (spring)
▪ Conservation of energy
▪ Power, units
Periodic Motion (PHY)
▪ Amplitude, frequency, phase
▪ Transverse and longitudinal waves: wavelength
and propagation speed

4B: Importance of fluids for the circulation of blood,
gas movement, and gas exchange
Fluids are featured in several physiologically important
processes, including the circulation of blood, gas
movement into and out of the lungs, and gas exchange
with the blood. The energetic requirements of fluid
dynamics can be modeled using physical equations. A
thorough understanding of fluids is necessary to
understand the origins of numerous forms of disease.

© 2020 Association of American Medical Colleges

59

Fluids (PHY)
▪ Density, specific gravity
▪ Buoyancy, Archimedes’ Principle
▪ Hydrostatic pressure
o Pascal’s Law
o Hydrostatic pressure; P = ρgh (pressure vs.
depth)
▪ Viscosity: Poiseuille Flow
▪ Continuity equation (A∙v = constant)
▪ Concept of turbulence at high velocities
▪ Surface tension
▪ Bernoulli’s equation

The content in this category covers hydrostatic
pressure, fluid flow rates, viscosity, the Kinetic
Molecular Theory of Gases, and the Ideal Gas Law.

▪ Venturi effect, pitot tube
Circulatory System (BIO)
▪ Arterial and venous systems; pressure and flow
characteristics
Gas Phase (GC, PHY)
▪ Absolute temperature, K, Kelvin scale
▪ Pressure, simple mercury barometer
▪ Molar volume at 0°C and 1 atm = 22.4 L/mol
▪ Ideal gas
o Definition
o Ideal Gas Law: PV = nRT
o Boyle’s Law: PV = constant
o Charles’ Law: V/T = constant
o Avogadro’s Law: V/n = constant
▪ Kinetic Molecular Theory of Gases
o Heat capacity at constant volume and at
constant pressure (PHY)
o Boltzmann’s Constant (PHY)
▪ Deviation of real gas behavior from Ideal Gas Law
o Qualitative
o Quantitative (Van der Waals’ Equation)
▪ Partial pressure, mole fraction
▪ Dalton’s Law relating partial pressure to
composition

4C: Electrochemistry and electrical circuits and their
elements
Charged particles can be set in motion by the action of
an applied electrical field and can be used to transmit
energy or information over long distances. The energy
released during certain chemical reactions can be
converted to electrical energy, which can be harnessed
to perform other reactions or work.
Physiologically, a concentration gradient of charged
particles is set up across the cell membrane of neurons
at considerable energetic expense. This allows for the

© 2020 Association of American Medical Colleges

60

Electrostatics (PHY)
▪ Charge, conductors, charge conservation
▪ Insulators
▪ Coulomb’s Law
▪ Electric field E
o Field lines
o Field due to charge distribution
▪ Electrostatic energy, electric potential at a point in
space

rapid transmission of signals using electrical
impulses — changes in the electrical voltage across
the membrane — under the action of some external
stimulus.
The content in this category covers electrical circuit
elements, electrical circuits, and electrochemistry.

Circuit Elements (PHY)
▪ Current I = ΔQ/Δt, sign conventions, units
▪ Electromotive force, voltage
▪ Resistance
o Ohm’s Law: I = V/R
o Resistors in series
o Resistors in parallel
o Resistivity: ρ = R•A/L
▪ Capacitance
o Parallel plate capacitor
o Energy of charged capacitor
o Capacitors in series
o Capacitors in parallel
o Dielectrics
▪ Conductivity
o Metallic
o Electrolytic
▪ Meters
Magnetism (PHY)
▪ Definition of magnetic field B
▪ Motion of charged particles in magnetic fields;
Lorentz force
Electrochemistry (GC)
▪ Electrolytic cell
o Electrolysis
o Anode, cathode
o Electrolyte
o Faraday’s Law relating amount of elements
deposited (or gas liberated) at an electrode to
current
o Electron flow; oxidation and reduction at the
electrodes
▪ Galvanic or Voltaic cells
o Half-reactions
o Reduction potentials; cell potential
o Direction of electron flow
▪ Concentration cell

© 2020 Association of American Medical Colleges

61

▪ Batteries
o Electromotive force, voltage
o Lead-storage batteries
o Nickel-cadmium batteries
Specialized Cell ― Nerve Cell (BIO)
▪ Myelin sheath, Schwann cells, insulation of axon
▪ Nodes of Ranvier: propagation of nerve impulse
along axon
4D: How light and sound interact with matter

Sound (PHY)

Light is a form of electromagnetic radiation — waves of
electric and magnetic fields that transmit energy. The
behavior of light depends on its frequency (or
wavelength). The properties of light are used in the
optical elements of the eye to focus rays of light on
sensory elements. When light interacts with matter,
spectroscopic changes occur that can be used to
identify the material on an atomic or molecular level.
Differential absorption of electromagnetic radiation
can be used to generate images useful in diagnostic
medicine. Interference and diffraction of light waves
are used in many analytical and diagnostic techniques.
The photon model of light explains why
electromagnetic radiation of different wavelengths
interacts differently with matter.

▪ Production of sound
▪ Relative speed of sound in solids, liquids, and
gases
▪ Intensity of sound, decibel units, log scale
▪ Attenuation (damping)
▪ Doppler Effect: moving sound source or observer,
reflection of sound from a moving object
▪ Pitch
▪ Resonance in pipes and strings
▪ Ultrasound
▪ Shock waves
Light, Electromagnetic Radiation (PHY)

▪ Concept of Interference; Young’s double-slit
experiment
▪ Thin films, diffraction grating, single-slit diffraction
When mechanical energy is transmitted through solids,
▪ Other diffraction phenomena, X-ray diffraction
liquids, and gases, oscillating pressure waves known as
▪ Polarization of light: linear and circular
“sound” are generated. Sound waves are audible if the
▪ Properties of electromagnetic radiation
sensory elements of the ear vibrate in response to
o Velocity equals constant c, in vacuo
exposure to these vibrations. The detection of reflected
o Electromagnetic radiation consists of
sound waves is used in ultrasound imaging. This
perpendicularly oscillating electric and magnetic
noninvasive technique readily locates dense
fields; direction of propagation is perpendicular
subcutaneous structures, such as bone and cartilage,
to both
and is very useful in diagnostic medicine.
▪ Classification of electromagnetic spectrum, photon
The content in this category covers the properties of
energy E = hf
both light and sound and how these energy waves
▪ Visual spectrum, color
interact with matter.
© 2020 Association of American Medical Colleges

62

Molecular Structure and Absorption Spectra (OC)
▪ Infrared region
o Intramolecular vibrations and rotations
o Recognizing common characteristic group
absorptions, fingerprint region
▪ Visible region (GC)
o Absorption in visible region gives
complementary color (e.g., carotene)
o Effect of structural changes on absorption (e.g.,
indicators)
▪ Ultraviolet region
o π-Electron and nonbonding electron transitions
o Conjugated systems
▪ NMR spectroscopy
o Protons in a magnetic field; equivalent protons
o Spin-spin splitting
Geometrical Optics (PHY)
▪ Reflection from plane surface: angle of incidence
equals angle of reflection
▪ Refraction, refractive index n; Snell’s law: n1 sin θ1
= n2 sin θ2
▪ Dispersion, change of index of refraction with
wavelength
▪ Conditions for total internal reflection
▪ Spherical mirrors
o Center of curvature
o Focal length
o Real and virtual images
▪ Thin lenses
o Converging and diverging lenses
o Use of formula 1/p + 1/q = 1/f, with sign
conventions
o Lens strength, diopters
▪ Combination of lenses
▪ Lens aberration
▪ Optical Instruments, including the human eye

© 2020 Association of American Medical Colleges

63

4E: Atoms, nuclear decay, electronic structure, and
atomic chemical behavior
Atoms are classified by their atomic number: the
number of protons in the atomic nucleus, which also
includes neutrons. Chemical interactions between
atoms are the result of electrostatic forces involving
the electrons and the nuclei. Because neutrons are
uncharged, they do not dramatically affect the
chemistry of any particular type of atom, but they do
affect the stability of the nucleus itself.
When a nucleus is unstable, decay results from one of
several different processes, which are random but
occur at well-characterized average rates. The products
of nuclear decay (alpha, beta, and gamma rays) can
interact with living tissue, breaking chemical bonds and
ionizing atoms and molecules in the process.
The electronic structure of an atom is responsible for
its chemical and physical properties. Only discrete
energy levels are allowed for electrons. These levels are
described individually by quantum numbers. Since the
outermost, or valence, electrons are responsible for the
strongest chemical interactions, a description of these
electrons alone is a good first approximation to
describe the behavior of any particular type of atom.
Mass spectrometry is an analytical tool that allows
characterization of atoms or molecules based on wellrecognized fragmentation patterns and the charge-tomass ratio (m/z) of ions generated in the gas phase.
The content in this category covers atomic structure,
nuclear decay, electronic structure, and the periodic
nature of atomic chemical behavior.

© 2020 Association of American Medical Colleges

64

Atomic Nucleus (PHY, GC)
▪ Atomic number, atomic weight
▪ Neutrons, protons, isotopes
▪ Nuclear forces, binding energy
▪ Radioactive decay
o α, β, γ decay
o Half-life, exponential decay, semi-log plots
▪ Mass spectrometer
▪ Mass spectroscopy
Electronic Structure (PHY, GC)
▪ Orbital structure of hydrogen atom, principal
quantum number n, number of electrons per
orbital (GC)
▪ Ground state, excited states
▪ Absorption and emission line spectra
▪ Use of Pauli Exclusion Principle
▪ Paramagnetism and diamagnetism
▪ Conventional notation for electronic structure (GC)
▪ Bohr atom
▪ Heisenberg Uncertainty Principle
▪ Effective nuclear charge (GC)
▪ Photoelectric effect
The Periodic Table ― Classification of Elements
Into Groups by Electronic Structure (GC)
▪ Alkali metals
▪ Alkaline earth metals: their chemical
characteristics
▪ Halogens: their chemical characteristics
▪ Noble gases: their physical and chemical
characteristics
▪ Transition metals
▪ Representative elements
▪ Metals and nonmetals
▪ Oxygen group

The Periodic Table ― Variations of Chemical
Properties with Group and Row (GC)
▪ Valence electrons
▪ First and second ionization energy
o Definition
o Prediction from electronic structure for
elements in different groups or rows
▪ Electron affinity
o Definition
o Variation with group and row
▪ Electronegativity
o Definition
o Comparative values for some representative
elements and important groups
▪ Electron shells and the sizes of atoms
▪ Electron shells and the sizes of ions
Stoichiometry (GC)
▪ Molecular weight
▪ Empirical vs. molecular formula
▪ Metric units commonly used in the context of
chemistry
▪ Description of composition by percent mass
▪ Mole concept, Avogadro’s number NA
▪ Definition of density
▪ Oxidation number
o Common oxidizing and reducing agents
o Disproportionation reactions
▪ Description of reactions by chemical equations
o Conventions for writing chemical equations
o Balancing equations, including redox equations
o Limiting reactants
o Theoretical yields

© 2020 Association of American Medical Colleges

65

Chemical and Physical Foundations of Biological Systems
Foundational Concept 5
The principles that govern chemical interactions and reactions form the basis for a broader understanding of
the molecular dynamics of living systems.
The chemical processes that take place within organisms are readily understood within the framework of the
behavior of solutions, thermodynamics, molecular structure, intermolecular interactions, molecular dynamics,
and molecular reactivity.
5A: Unique nature of water and its solutions

Acid-Base Equilibria (GC, BC)

To fully understand the complex and dynamic nature
▪ Brønsted-Lowry definition of acid, base
of living systems, it is first necessary to understand the ▪ Ionization of water
unique nature of water and its solutions. The unique
o Kw, its approximate value (Kw = [H+][OH–] = 10–14
properties of water allow it to strongly interact with
at 25°C, 1 atm)
and mobilize many types of solutes, including ions.
o Definition of pH: pH of pure water
Water is also unique in its ability to absorb energy and ▪ Conjugate acids and bases (e.g., NH4+ and NH3)
buffer living systems from the chemical changes
▪ Strong acids and bases (e.g., nitric, sulfuric)
necessary to sustain life.
▪ Weak acids and bases (e.g., acetic, benzoic)
o Dissociation of weak acids and bases with or
The content in this category covers the nature of
without added salt
solutions, solubility, acids, bases, and buffers.
o Hydrolysis of salts of weak acids or bases
o Calculation of pH of solutions of salts of weak
acids or bases
▪ Equilibrium constants Ka and Kb: pKa, pKb
▪ Buffers
o Definition and concepts (common buffer systems)
o Influence on titration curves
Ions in Solutions (GC, BC)
▪ Anion, cation: common names, formulas, and
charges for familiar ions (e.g., NH4+ ammonium,
PO43– phosphate, SO42– sulfate)
▪ Hydration, the hydronium ion

© 2020 Association of American Medical Colleges

66

Solubility (GC)
▪ Units of concentration (e.g., molarity)
▪ Solubility product constant; the equilibrium
expression Ksp
▪ Common-ion effect, its use in laboratory
separations
o Complex ion formation
o Complex ions and solubility
o Solubility and pH
Titration (GC)
▪ Indicators
▪ Neutralization
▪ Interpretation of the titration curves
▪ Redox titration
5B: Nature of molecules and intermolecular
interactions

Covalent Bond (GC)

Covalent bonding involves the sharing of electrons
between atoms. If the result of such interactions is not
a network solid, then the covalently bonded substance
will be discrete and molecular.
The shape of molecules can be predicted based on
electrostatic principles and quantum mechanics since
only two electrons can occupy the same orbital. Bond
polarity (both direction and magnitude) can be
predicted based on knowledge of the valence electron
structure of the constituent atoms. The strength of
intermolecular interactions depends on molecular
shape and the polarity of the covalent bonds present.
The solubility and other physical properties of
molecular substances depend on the strength of
intermolecular interactions.
The content in this category covers the nature of
molecules and includes covalent bonding, molecular
structure, nomenclature, and intermolecular
interactions.

© 2020 Association of American Medical Colleges

67

▪ Lewis electron dot formulas
o Resonance structures
o Formal charge
o Lewis acids and bases
▪ Partial ionic character
o Role of electronegativity in determining charge
distribution
o Dipole moment
▪ σ and π bonds
o Hybrid orbitals: sp3, sp2, sp, and respective
geometries
o Valence shell electron pair repulsion and the
prediction of shapes of molecules (e.g., NH3, H2O,
CO2)
o Structural formulas for molecules involving H, C,
N, O, F, S, P, Si, Cl
o Delocalized electrons and resonance in ions and
molecules
▪ Multiple bonding
o Effect on bond length and bond energies
o Rigidity in molecular structure

▪ Stereochemistry of covalently bonded molecules
(OC)
o Isomers
▪ Structural isomers
▪ Stereoisomers (e.g., diastereomers,
enantiomers, cis-trans isomers)
▪ Conformational isomers
o Polarization of light, specific rotation
o Absolute and relative configuration
▪ Conventions for writing R and S forms
▪ Conventions for writing E and Z forms
Liquid Phase ― Intermolecular Forces (GC)
▪ Hydrogen bonding
▪ Dipole Interactions
▪ Van der Waals’ Forces (London dispersion forces)
5C: Separation and purification methods

Separations and Purifications (OC, BC)

Analysis of complex mixtures of substances ―
especially biologically relevant materials ― typically
requires separation of the components. Many
methods have been developed to accomplish this
task, and the method used is dependent on the types
of substances which comprise the mixture. All these
methods rely on the magnification of potential
differences in the strength of intermolecular
interactions.

▪ Extraction: distribution of solute between two
immiscible solvents
▪ Distillation
▪ Chromatography: basic principles involved in
separation process
o Column chromatography
▪ Gas-liquid chromatography
▪ High-pressure liquid chromatography
o Paper chromatography
o Thin-layer chromatography
▪ Separation and purification of peptides and
proteins (BC)
o Electrophoresis
o Quantitative analysis
o Chromatography
▪ Size-exclusion
▪ Ion-exchange
▪ Affinity
▪ Racemic mixtures, separation of enantiomers (OC)

The content in this category covers separation and
purification methods including extraction, liquid and
gas chromatography, and electrophoresis.

© 2020 Association of American Medical Colleges

68

5D: Structure, function, and reactivity of biologically
relevant molecules
The structure of biological molecules forms the basis
of their chemical reactions including oligomerization
and polymerization. Unique aspects of each type of
biological molecule dictate their role in living systems,
whether providing structure or information storage or
serving as fuel and catalysts.
The content in this category covers the structure,
function, and reactivity of biologically relevant
molecules including the mechanistic considerations
that dictate their modes of reactivity.

Nucleotides and Nucleic Acids (BC, BIO)
▪ Nucleotides and nucleosides: composition
o Sugar phosphate backbone
o Pyrimidine, purine residues
▪ Deoxyribonucleic acid: DNA; ribonucleic acid: RNA;
double helix; RNA structures
▪ Chemistry (BC)
▪ Other functions (BC)
Amino Acids, Peptides, Proteins (OC, BC)
▪ Amino acids: description
o Absolute configuration at the α position
o Dipolar ions
o Classification
▪ Acidic or basic
▪ Hydrophilic or hydrophobic
o Synthesis of α-amino acids (OC)
▪ Strecker Synthesis
▪ Gabriel Synthesis
▪ Peptides and proteins: reactions
o Sulfur linkage for cysteine and cystine
o Peptide linkage: polypeptides and proteins
o Hydrolysis (BC)
▪ General principles
o Primary structure of proteins
o Secondary structure of proteins
o Tertiary structure of proteins
o Isoelectric point
The Three-Dimensional Protein Structure (BC)
▪ Conformational stability
o Hydrophobic interactions
o Solvation layer (entropy)
▪ Quaternary structure
▪ Denaturing and folding

© 2020 Association of American Medical Colleges

69

Nonenzymatic Protein Function (BC)
▪ Binding
▪ Immune system
▪ Motor
Lipids (BC, OC)
▪ Description, types
o Storage
▪ Triacyl glycerols
▪ Free fatty acids: saponification
o Structural
▪ Phospholipids and phosphatids
▪ Sphingolipids (BC)
▪ Waxes
o Signals, cofactors
▪ Fat-soluble vitamins
▪ Steroids
▪ Prostaglandins (BC)
Carbohydrates (OC)
▪ Description
o Nomenclature and classification, common names
o Absolute configuration
o Cyclic structure and conformations of hexoses
o Epimers and anomers
▪ Hydrolysis of the glycoside linkage
▪ Keto-enol tautomerism of monosaccharides
▪ Disaccharides (BC)
▪ Polysaccharides (BC)
Aldehydes and Ketones (OC)
▪ Description
o Nomenclature
o Physical properties
▪ Important reactions
o Nucleophilic addition reactions at C=O bond
▪ Acetal, hemiacetal
▪ Imine, enamine
▪ Hydride reagents
© 2020 Association of American Medical Colleges

70

▪ Cyanohydrin
o Oxidation of aldehydes
o Reactions at adjacent positions: enolate
chemistry
▪ Keto-enol tautomerism (α-racemization)
▪ Aldol condensation, retro-aldol
▪ Kinetic vs. thermodynamic enolate
▪ General principles
o Effect of substituents on reactivity of C=O; steric
hindrance
o Acidity of α-H; carbanions
Alcohols (OC)
▪ Description
o Nomenclature
o Physical properties (acidity, hydrogen bonding)
▪ Important reactions
o Oxidation
o Substitution reactions: SN1 or SN2
o Protection of alcohols
o Preparation of mesylates and tosylates
Carboxylic Acids (OC)
▪ Description
o Nomenclature
o Physical properties
▪ Important reactions
o Carboxyl group reactions
▪ Amides (and lactam), esters (and lactone),
anhydride formation
▪ Reduction
▪ Decarboxylation
o Reactions at 2-position, substitution
Acid Derivatives (Anhydrides, Amides, Esters) (OC)
▪ Description
o Nomenclature
o Physical properties
▪ Important reactions
o Nucleophilic substitution
© 2020 Association of American Medical Colleges

71

o Transesterification
o Hydrolysis of amides
▪ General principles
o Relative reactivity of acid derivatives
o Steric effects
o Electronic effects
o Strain (e.g., β-lactams)
Phenols (OC, BC)
▪ Oxidation and reduction (e.g., hydroquinones,
ubiquinones): biological 2e– redox centers
Polycyclic and Heterocyclic Aromatic Compounds
(OC, BC)
▪ Biological aromatic heterocycles
5E: Principles of chemical thermodynamics and
kinetics

Enzymes (BC, BIO)

The processes that occur in living systems are
dynamic, and they follow the principles of chemical
thermodynamics and kinetics. The position of
chemical equilibrium is dictated by the relative
energies of products and reactants. The rate at which
chemical equilibrium is attained is dictated by a
variety of factors: concentration of reactants,
temperature, and the amount of catalyst (if any).
Biological systems have evolved to harness energy and
use it in very efficient ways to support all processes of
life, including homeostasis and anabolism. Biological
catalysts, known as enzymes, have evolved that allow
all the relevant chemical reactions required to sustain
life to occur both rapidly and efficiently and under the
narrow set of conditions required.
The content in this category covers all principles of
chemical thermodynamics and kinetics including
enzymatic catalysis.

© 2020 Association of American Medical Colleges

72

▪ Classification by reaction type
▪ Mechanism
o Substrates and enzyme specificity
o Active-site model
o Induced-fit model
o Cofactors, coenzymes, and vitamins
▪ Kinetics
o General (catalysis)
o Michaelis-Menten
o Cooperativity
o Effects of local conditions on enzyme activity
▪ Inhibition
▪ Regulatory enzymes
o Allosteric
o Covalently modified
Principles of Bioenergetics (BC)
▪ Bioenergetics/thermodynamics
o Free energy, Keq
o Concentration
▪ Phosphorylation/ATP
o ATP hydrolysis ΔG << 0

o ATP group transfers
▪ Biological oxidation-reduction
o Half-reactions
o Soluble electron carriers
o Flavoproteins
Energy Changes in Chemical Reactions ―
Thermochemistry, Thermodynamics (GC, PHY)
▪ Thermodynamic system – state function
▪ Zeroth Law – concept of temperature
▪ First Law − conservation of energy in
thermodynamic processes
▪ PV diagram: work done = area under or enclosed by
curve (PHY)
▪ Second Law – concept of entropy
o Entropy as a measure of “disorder”
o Relative entropy for gas, liquid, and crystal states
▪ Measurement of heat changes (calorimetry), heat
capacity, specific heat
▪ Heat transfer – conduction, convection, radiation
(PHY)
▪ Endothermic, exothermic reactions (GC)
o Enthalpy, H, and standard heats of reaction and
formation
o Hess’ Law of Heat Summation
▪ Bond dissociation energy as related to heats of
formation (GC)
▪ Free energy: G (GC)
▪ Spontaneous reactions and ΔG° (GC)
▪ Coefficient of expansion (PHY)
▪ Heat of fusion, heat of vaporization
▪ Phase diagram: pressure and temperature
Rate Processes in Chemical Reactions ― Kinetics and
Equilibrium (GC)
▪ Reaction rate
▪ Dependence of reaction rate on concentration of
reactants
o Rate law, rate constant
o Reaction order
© 2020 Association of American Medical Colleges

73

▪ Rate-determining step
▪ Dependence of reaction rate on temperature
o Activation energy
▪ Activated complex or transition state
▪ Interpretation of energy profiles showing
energies of reactants, products, activation
energy, and ΔH for the reaction
o Use of the Arrhenius Equation
▪ Kinetic control vs. thermodynamic control of a
reaction
▪ Catalysts
▪ Equilibrium in reversible chemical reactions
o Law of Mass Action
o Equilibrium Constant
o Application of Le Châtelier’s Principle
▪ Relationship of the equilibrium constant and ΔG°

© 2020 Association of American Medical Colleges

74

Psychological, Social, and Biological Foundations of Behavior
What Will the Psychological, Social, and Biological Foundations of Behavior Section Test?
The Psychological, Social, and Biological Foundations of Behavior section asks you to solve problems by
combining your knowledge of foundational concepts with your scientific inquiry and reasoning skills.
This section tests your understanding of the ways psychological, social, and biological factors influence
perceptions and reactions to the world; behavior and behavior change; what people think about
themselves and others; the cultural and social differences that influence well-being; and the
relationships between social stratification, access to resources, and well-being.
The Psychological, Social, and Biological Foundations of Behavior section emphasizes concepts that
tomorrow’s doctors need to know in order to serve an increasingly diverse population and have a clear
understanding of the impact of behavior on health. Further, it communicates the need for future
physicians to be prepared to deal with the human and social issues of medicine.
This section is designed to:
▪
▪
▪
▪
▪

Test psychology, sociology, and biology concepts that provide a solid foundation for learning in
medical school about the behavioral and sociocultural determinants of health.
Test concepts taught at many colleges and universities in first-semester psychology and
sociology courses.
Test biology concepts that relate to mental processes and behavior taught at many colleges and
universities in introductory biology.
Test basic research methods and statistics concepts described by many baccalaureate faculty as
important to success in introductory science courses.
Require you to demonstrate your scientific inquiry and reasoning, research methods, and
statistics skills as applied to the social and behavioral sciences.

Test Section

Number of Questions

Time

Psychological, Social, and
Biological Foundations of
Behavior

59

95 minutes

(note that questions are a
combination of passage-based
and discrete questions)

© 2020 Association of American Medical Colleges

75

Scientific Inquiry and Reasoning Skills
As a reminder, the scientific inquiry and reasoning skills you will be asked to demonstrate on this section
of the exam are:
Knowledge of Scientific Concepts and Principles
▪
▪

Demonstrating understanding of scientific concepts and principles.
Identifying the relationships between closely related concepts.

Scientific Reasoning and Problem-Solving
▪
▪

Reasoning about scientific principles, theories, and models.
Analyzing and evaluating scientific explanations and predictions.

Reasoning About the Design and Execution of Research
▪
▪

Demonstrating understanding of important components of scientific research.
Reasoning about ethical issues in research.

Data-Based and Statistical Reasoning
▪
▪

Interpreting patterns in data presented in tables, figures, and graphs.
Reasoning about data and drawing conclusions from them.

© 2020 Association of American Medical Colleges

76

General Mathematical Concepts and Techniques
It’s important for you to know that questions on the natural, behavioral, and social sciences sections will ask you
to use certain mathematical concepts and techniques. As the descriptions of the scientific inquiry and reasoning
skills suggest, some questions will ask you to analyze and manipulate scientific data to show you can:
▪
▪
▪

▪
▪
▪

▪

Recognize and interpret linear, semilog, and log-log scales and calculate slopes from data found in figures,
graphs, and tables.
Demonstrate a general understanding of significant digits and the use of reasonable numerical estimates
in performing measurements and calculations.
Use metric units, including converting units within the metric system and between metric and English
units (conversion factors will be provided when needed), and dimensional analysis (using units to balance
equations).
Perform arithmetic calculations involving the following: probability, proportion, ratio, percentage, and
square-root estimations.
Demonstrate a general understanding (Algebra II-level) of exponentials and logarithms (natural and base
10), scientific notation, and solving simultaneous equations.
Demonstrate a general understanding of the following trigonometric concepts: definitions of basic (sine,
cosine, tangent) and inverse (sin‒1, cos‒1, tan‒1) functions; sin and cos values of 0°, 90°, and 180°;
relationships between the lengths of sides of right triangles containing angles of 30°, 45°, and 60°.
Demonstrate a general understanding of vector addition and subtraction and the right-hand rule
(knowledge of dot and cross products is not required)

Note also that an understanding of calculus is not required, and a periodic table will be provided during the exam.

Psychological, Social, and Biological Foundations of Behavior Distribution of Questions by
Discipline, Foundational Concept, and Scientific Inquiry and Reasoning Skill
You may wonder how much psychology, sociology, and biology you’ll see on this section of the MCAT
exam, how many questions you’ll get about a particular foundational concept, or how the scientific
inquiry and reasoning skills will be distributed on your exam. The questions you see are likely to be
distributed in the ways described below. These are the approximate percentages of questions you’ll see
on a test for each discipline, foundational concept, and scientific inquiry and reasoning skill.*

* Please note that about 5% of this test section will include psychology questions that are biologically
relevant. This is in addition to the discipline target of 5% for introductory biology specified for this
section.
© 2020 Association of American Medical Colleges

77

(These percentages have been approximated to the nearest 5% and will vary from one test to another
for a variety of reasons, including, but not limited to, controlling for question difficulty, using groups of
questions that depend on a single passage, and using unscored field-test questions on each test form.)
Discipline:
▪
▪
▪

Introductory psychology, 65%
Introductory sociology, 30%
Introductory biology, 5%

Foundational Concept:
▪
▪
▪
▪
▪

Foundational Concept 6, 25%
Foundational Concept 7, 35%
Foundational Concept 8, 20%
Foundational Concept 9, 15%
Foundational Concept 10, 5%

Scientific Inquiry and Reasoning Skill:
▪
▪
▪
▪

Skill 1, 35%
Skill 2, 45%
Skill 3, 10%
Skill 4: 10%

Psychological, Social, and Biological Foundations of Behavior Framework of Foundational
Concepts and Content Categories
Foundational Concept 6: Biological, psychological, and sociocultural factors influence the ways that
individuals perceive, think about, and react to the world.
The content categories for this foundational concept include
6A. Sensing the environment
6B. Making sense of the environment
6C. Responding to the world
Foundational Concept 7: Biological, psychological, and sociocultural factors influence behavior and
behavior change.
The content categories for this foundational concept include
7A. Individual influences on behavior
7B. Social processes that influence human behavior

© 2020 Association of American Medical Colleges

78

7C. Attitude and behavior change
Foundational Concept 8: Psychological, sociocultural, and biological factors influence the way we think
about ourselves and others, as well as how we interact with others.
The content categories for this foundational concept include
8A. Self-identity
8B. Social thinking
8C. Social interactions
Foundational Concept 9: Cultural and social differences influence well-being.
The content categories for this foundational concept include
9A. Understanding social structure
9B. Demographic characteristics and processes
Foundational Concept 10: Social stratification and access to resources influence well-being.
The content category for this foundational concept is
10A. Social inequality

How Foundational Concepts and Content Categories Fit Together
The MCAT exam asks you to solve problems by combining your knowledge of concepts with your
scientific inquiry and reasoning skills. The figure below illustrates how foundational concepts, content
categories, and scientific inquiry and reasoning skills intersect to create test questions.

Foundational Concept 1

Skill

Content
Category 1A

Foundational Concept 2

Content
Category 1B

Skill 1

▪

Skill 2
Skill 3
Skill 4

▪

Content
Category 1C

Content
Category 2A

Content
Category 2B

Content
Category 2C

Each cell represents the point at which foundational
concepts, content categories, and scientific inquiry and
reasoning skills cross.
Test questions are written at the intersections of the
knowledge and skills.

© 2020 Association of American Medical Colleges

79

Understanding the Foundational Concepts and Content Categories in the Psychological,
Social, and Biological Foundations of Behavior Section
The following are detailed explanations of each foundational concept and related content category
tested by the Psychological, Social, and Biological Foundational of Behavior section. As with the natural
sciences sections, content lists describing specific topics and subtopics that define each content category
are provided. The same content list is provided to the writers who develop the content of the exam.
Here is an excerpt from the content list.
EXCERPT FROM THE PSYCHOLOGICAL, SOCIAL, AND BIOLOGICAL FOUNDATONS OF BEHAVIOR OUTLINE
Self-Presentation and Interacting With Others (PSY, SOC)
▪

▪

▪
▪

Topic

Expressing and detecting emotion
Subtopic
o The role of gender in the expression and detection of emotion
o The role of culture in the expression and detection of emotion
Presentation of self
o Impression management
o Front-stage vs. back-stage self (dramaturgical approach) (SOC)
Verbal and nonverbal communication
Animal signals and communication (PSY, BIO)

The abbreviations found in parentheses indicate the course(s) in which undergraduate students at many
colleges and universities learn about the topics and associated subtopics. The course abbreviations are:
▪
▪
▪

PSY: one semester of introductory psychology
SOC: one semester of introductory sociology
BIO: two-semester sequence of introductory biology

In preparing for the MCAT exam, you will be responsible for learning the topics and associated subtopics
at the levels taught in the courses listed in parentheses. A small number of subtopics have course
abbreviations indicated in parentheses. In those cases, you are responsible only for learning the
subtopics as they are taught in the course(s) indicated.
Using the excerpt above as an example:
▪

▪

You are responsible for learning about the topic Self-Presentation and Interacting With Others
at the level taught in a typical introductory psychology course and in a typical introductory
sociology course.
You are responsible for learning about the sub-subtopic Front-stage vs. back-stage self
(dramaturgical approach) only at the level taught in a typical introductory sociology course.

© 2020 Association of American Medical Colleges

80

▪

You are responsible for learning about the subtopic Animal signals and communication at the
level taught in a typical introductory psychology course and in a typical introductory biology
course.

Remember that course content at your school may differ from course content at other colleges and
universities. The topics and subtopics described in this chapter may be covered in courses with titles
that are different from those listed here. Your prehealth advisor and faculty are important resources for
your questions about course content.

© 2020 Association of American Medical Colleges

81

Psychological, Social, and Biological Foundations of Behavior
Foundational Concept 6
Biological, psychological, and sociocultural factors influence the ways that individuals perceive, think about, and
react to the world.
The way we sense, perceive, think about, and react to stimuli affects our experiences. Foundational Concept 6
focuses on these components of experience, starting with the initial detection and perception of stimuli through
cognition and continuing to emotion and stress.
6A: Sensing the environment

Sensory Processing (PSY, BIO)

Psychological, sociocultural, and biological factors
affect how we sense and perceive the world. All
sensory processing begins with first detecting a
stimulus in the environment through sensory cells,
receptors, and biological pathways.

▪ Sensation
o Threshold
o Weber’s Law (PSY)
o Signal detection theory (PSY)
o Sensory adaptation
o Psychophysics
▪ Sensory receptors
o Sensory pathways
o Types of sensory receptors

After collecting sensory information, we then
interpret and make sense of it. Although sensation
and perception are distinct functions, they are both
influenced by psychological, social, and biological
factors and thus become almost indistinguishable in
practice. This complexity is illuminated by examining
human sight, hearing, touch, taste, and smell.
The content in this category covers sensation and
perception across all human senses.

Vision (PSY, BIO)
▪ Structure and function of the eye
▪ Visual processing
o Visual pathways in the brain
o Parallel processing (PSY)
o Feature detection (PSY)
Hearing (PSY, BIO)
▪ Structure and function of the ear
▪ Auditory processing (e.g., auditory pathways in the
brain)
▪ Sensory reception by hair cells
Other Senses (PSY, BIO)
▪ Somatosensation (e.g., pain perception)
▪ Taste (e.g., taste buds (chemoreceptors) that detect
specific chemicals)

© 2020 Association of American Medical Colleges

82

▪ Smell
o Olfactory cells (chemoreceptors) that detect
specific chemicals
o Pheromones (BIO)
o Olfactory pathways in the brain (BIO)
▪ Kinesthetic sense (PSY)
▪ Vestibular sense
Perception (PSY)
▪ Bottom-up/top-down processing
▪ Perceptual organization (e.g., depth, form, motion,
constancy)
▪ Gestalt principles
6B: Making sense of the environment

Attention (PSY)

The way we think about the world depends on our
awareness, thoughts, knowledge, and memories. It is
also influenced by our ability to solve problems, make
decisions, form judgments, and communicate.
Psychological, sociocultural, and biological influences
determine the development and use of these different
yet convergent processes.

▪ Selective attention
▪ Divided attention

Biological factors underlie the mental processes that
create our reality, shape our perception of the world,
and influence the way we perceive and react to every
aspect of our lives.
The content in this category covers critical aspects of
cognition ― including consciousness, cognitive
development, problem-solving and decision-making,
intelligence, memory, and language.

© 2020 Association of American Medical Colleges

83

Cognition (PSY)
▪ Information-processing model
▪ Cognitive development
o Piaget’s stages of cognitive development
o Cognitive changes in late adulthood
o Role of culture in cognitive development
o Influence of heredity and environment on cognitive
development
▪ Biological factors that affect cognition (PSY, BIO)
▪ Problem-solving and decision-making
o Types of problem-solving
o Barriers to effective problem-solving
o Approaches to problem-solving
o Heuristics and biases (e.g., overconfidence, belief
perseverance)
▪ Intellectual functioning
o Theories of intelligence
o Influence of heredity and environment on
intelligence
o Variations in intellectual ability

Consciousness (PSY)
▪ States of consciousness
o Alertness (PSY, BIO)
o Sleep
▪ Stages of sleep
▪ Sleep cycles and changes to sleep cycles
▪ Sleep and circadian rhythms (PSY, BIO)
▪ Dreaming
▪ Sleep-wake disorders
o Hypnosis and meditation
▪ Consciousness-altering drugs
o Types of consciousness-altering drugs and their
effects on the nervous system and behavior
o Drug addiction and the reward pathway in the brain
Memory (PSY)
▪ Encoding
o Process of encoding information
o Processes that aid in encoding memories
▪ Storage
o Types of memory storage (e.g., sensory, working,
long-term)
o Semantic networks and spreading activation
▪ Retrieval
o Recall, recognition, and relearning
o Retrieval cues
o The role of emotion in retrieving memories (PSY,
BIO)
o Processes that aid retrieval
▪ Forgetting
o Aging and memory
o Memory dysfunctions (e.g., Alzheimer’s disease,
Korsakoff’s syndrome)
o Decay
o Interference
o Memory construction and source monitoring
▪ Changes in synaptic connections underlie memory
and learning (PSY, BIO)
o Neural plasticity

© 2020 Association of American Medical Colleges

84

o Memory and learning
o Long-term potentiation
Language (PSY)
▪ Theories of language development (e.g., learning,
nativist, interactionist)
▪ Influence of language on cognition
▪ Brain areas that control language and speech (PSY,
BIO)
6C: Responding to the world

Emotion (PSY)

We experience a barrage of environmental stimuli
throughout the course of our lives. In many cases,
environmental stimuli trigger physiological responses,
such as an elevated heart rate, increased perspiration,
or heightened feelings of anxiety. How we perceive
and interpret these physiological responses is complex
and influenced by psychological, sociocultural, and
biological factors.

▪ Three components of emotion (i.e., cognitive,
physiological, behavioral)
▪ Universal emotions (i.e., fear, anger, happiness,
surprise, joy, disgust, sadness)
▪ Adaptive role of emotion
▪ Theories of emotion
o James-Lange theory
o Cannon-Bard theory
o Schachter-Singer theory
▪ The role of biological processes in perceiving emotion
(PSY, BIO)
o Brain regions involved in the generation and
experience of emotions
o The role of the limbic system in emotion
o Emotion and the autonomic nervous system
o Physiological markers of emotion (signatures of
emotion)

Emotional responses, such as feelings of happiness,
sadness, anger, or stress, are often born out of our
interpretation of this interplay of physiological
responses. Our experience with emotions and stress
not only affects our behavior, but also shapes our
interactions with others.
The content in this category covers the basic
components and theories of emotion and their
underlying psychological, sociocultural, and biological
factors. It also addresses stress, stress outcomes, and
stress management.

© 2020 Association of American Medical Colleges

85

Stress (PSY)
▪ The nature of stress
o Appraisal
o Different types of stressors (e.g., cataclysmic
events, personal)
o Effects of stress on psychological functions

▪ Stress outcomes, response to stressors
o Physiological (PSY, BIO)
o Emotional
o Behavioral
▪ Managing stress (e.g., exercise, relaxation,
spirituality)

© 2020 Association of American Medical Colleges

86

Psychological, Social, and Biological Foundations of Behavior
Foundational Concept 7
Biological, psychological, and sociocultural factors influence behavior and behavior change.
Human behavior is complex and often surprising, differing across individuals in the same situation and within an
individual across different situations. A full understanding of human behavior requires knowledge of the
interplay between psychological, sociocultural, and biological factors related to behavior. This interplay has
important implications for the way we behave and the likelihood of behavior change.
Foundational Concept 7 focuses on individual and social determinants of behavior and behavior change.
Content Categories
▪
▪
▪

Category 7A focuses on the individual psychological and biological factors that affect behavior.
Category 7B focuses on how social factors, such as groups and social norms, affect behavior.
Category 7C focuses on how learning affects behavior, as well as the role of attitude theories in behavior
and behavior change.

With these building blocks, medical students will be able to learn how behavior can either support health or
increase risk for disease.
7A: Individual influences on behavior

Biological Bases of Behavior (PSY, BIO)

A complex interplay of psychological and biological
factors shapes behavior. Biological structures and
processes serve as the pathways by which bodies
carry out activities. They also affect predispositions to
behave in certain ways, shape personalities, and
influence the likelihood of developing psychological
disorders. Psychological factors also affect behavior
and, consequently, health and well-being.

▪ The nervous system
o Neurons (e.g., the reflex arc)
o Neurotransmitters
o Structure and function of the peripheral nervous
system
o Structure and function of the central nervous
system
▪ The brain
o Forebrain
o Midbrain
o Hindbrain
o Lateralization of cortical functions
o Methods used in studying the brain
▪ The spinal cord
▪ Neuronal communication and its influence on
behavior (PSY)
▪ Influence of neurotransmitters on behavior (PSY)

The content in this category covers biological bases of
behavior, including the effect of genetics and how the
nervous and endocrine systems affect behavior. It also
addresses how personality, psychological disorders,
motivation, and attitudes affect behavior. Some of
these topics are learned in the context of nonhuman
animal species.

© 2020 Association of American Medical Colleges

87

▪ The endocrine system
o Components of the endocrine system
o Effects of the endocrine system on behavior
▪ Behavioral genetics
o Genes, temperament, and heredity
o Adaptive value of traits and behaviors
o Interaction between heredity and environmental
influences
▪ Influence of genetic and environmental factors on the
development of behaviors
o Experience and behavior (PSY)
o Regulatory genes and behavior (BIO)
o Genetically based behavioral variation in natural
populations
▪ Human physiological development (PSY)
o Prenatal development
o Motor development
o Developmental changes in adolescence
Personality (PSY)
▪ Theories of personality
o Psychoanalytic perspective
o Humanistic perspective
o Trait perspective
o Social cognitive perspective
o Biological perspective
o Behaviorist perspective
▪ Situational approach to explaining behavior
Psychological Disorders (PSY)
▪ Understanding psychological disorders
o Biomedical vs. biopsychosocial approaches
o Classifying psychological disorders
o Rates of psychological disorders
▪ Types of psychological disorders
o Anxiety disorders
o Obsessive-compulsive disorder
o Trauma- and stressor-related disorders
o Somatic symptom and related disorders
o Bipolar and related disorders
© 2020 Association of American Medical Colleges

88

o Depressive disorders
o Schizophrenia
o Dissociative disorders
o Personality disorders
▪ Biological bases of nervous system disorders (PSY,
BIO)
o Schizophrenia
o Depression
o Alzheimer’s disease
o Parkinson’s disease
o Stem cell-based therapy to regenerate neurons in
the central nervous system (BIO)
Motivation (PSY)
▪ Factors that influence motivation
o Instinct
o Arousal
o Drives (e.g., negative-feedback systems) (PSY, BIO)
o Needs
▪ Theories that explain how motivation affects human
behavior
o Drive reduction theory
o Incentive theory
o Other theories (e.g., cognitive, need-based)
▪ Biological and sociocultural motivators that regulate
behavior (e.g., hunger, sex drive, substance addiction)
Attitudes (PSY)
▪ Components of attitudes (i.e., cognitive, affective,
behavioral)
▪ The link between attitudes and behavior
o Processes by which behavior influences attitudes
(e.g., foot-in-the door phenomenon, role-playing
effects)
o Processes by which attitudes influence behavior
o Cognitive dissonance theory

© 2020 Association of American Medical Colleges

89

7B: Social processes that influence human behavior
Many social processes influence human behavior; in
fact, the mere presence of other individuals can
influence our behavior. Groups and social norms also
exert influence over our behavior. Oftentimes, social
processes influence our behavior through unwritten
rules that define acceptable and unacceptable
behavior in society.
Our understanding of groups and social norms is
learned through the process of socialization. What we
learn about the groups and society to which we
belong affects our behavior and influences our
perceptions and interactions with others.
The content in this category covers how the presence
of others, group decision-making processes, social
norms, and socialization shape our behavior.

How the Presence of Others Affects Individual
Behavior (PSY)
▪ Social facilitation
▪ Deindividuation
▪ Bystander effect
▪ Social loafing
▪ Social control (SOC)
▪ Peer pressure (PSY, SOC)
▪ Conformity (PSY, SOC)
▪ Obedience (PSY, SOC)
Group Decision-Making Processes (PSY, SOC)
▪ Group polarization (PSY)
▪ Groupthink
Normative and Nonnormative Behavior (SOC)
▪ Social norms (PSY, SOC)
o Sanctions (SOC)
o Folkways, mores, and taboos (SOC)
o Anomie (SOC)
▪ Deviance
o Perspectives on deviance (e.g., differential
association, labeling theory, strain theory)
▪ Aspects of collective behavior (e.g., fads, mass
hysteria, riots)
Socialization (PSY, SOC)
▪ Agents of socialization (e.g., the family, mass media,
peers, workplace)

7C: Attitude and behavior change

Habituation and Dishabituation (PSY)

Learning is a relatively permanent change in behavior
brought about by experience. There are a number of
different types of learning, which include habituation
as well as associative, observational, and social
learning.

Associative Learning (PSY)

Although people can learn new behaviors and change
their attitudes, psychological, environmental, and
© 2020 Association of American Medical Colleges

90

▪ Classical conditioning (PSY, BIO)
o Neutral, conditioned, and unconditioned stimuli
o Conditioned and unconditioned response
o Processes: acquisition, extinction, spontaneous
recovery, generalization, discrimination

biological factors influence whether those changes will
be short-term or long-term. Understanding how
people learn new behaviors and change their attitudes
and which conditions affect learning helps us
understand behavior and our interactions with others.
The content in this category covers learning and
theories of attitude and behavior change. This
includes the elaboration likelihood model and social
cognitive theory.

▪ Operant conditioning (PSY, BIO)
o Processes of shaping and extinction
o Types of reinforcement: positive, negative, primary,
conditional
o Reinforcement schedules: fixed-ratio, variableratio, fixed-interval, variable-interval
o Punishment
o Escape and avoidance learning
▪ The role of cognitive processes in associative learning
▪ Biological processes that affect associative learning
(e.g., biological predispositions, instinctive drift) (PSY,
BIO)
Observational Learning (PSY)
▪ Modeling
▪ Biological processes that affect observational learning
o Mirror neurons
o Role of the brain in experiencing vicarious emotions
▪ Applications of observational learning to explain
individual behavior
Theories of Attitude and Behavior Change (PSY)
▪ Elaboration likelihood model
▪ Social cognitive theory
▪ Factors that affect attitude change (e.g., changing
behavior, characteristics of the message and target,
social factors)

© 2020 Association of American Medical Colleges

91

Psychological, Social, and Biological Foundations of Behavior
Foundational Concept 8
Psychological, sociocultural, and biological factors influence the way we think about ourselves and others, as
well as how we interact with others.
The connection between how people think about themselves and others is complex and affects social
interactions. The interplay between thoughts about ourselves, thoughts about others, and our biology has
important implications for our sense of self and interpersonal relationships.
Foundational Concept 8 focuses on the physical, cognitive, and social components of our identity, as well as
how these components influence the way we think about and interact with others.
Content Categories
▪
▪
▪

Category 8A focuses on the notion of self and identity formation.
Category 8B focuses on the attitudes and beliefs that affect social interaction.
Category 8C focuses on the actions and processes underlying social interactions.

With these building blocks, medical students will be able to learn how to communicate and collaborate with
patients and other members of the health care team.
8A: Self-identity
The self refers to the thoughts and beliefs we have
about ourselves. Our notion of the self is complex and
multifaceted. It includes gender, racial, and ethnic
identities, as well as beliefs about our ability to
accomplish tasks and exert control over different
situations.
Our notion of the self develops over time and is
shaped by a variety of factors, including society,
culture, individuals and groups, and our unique
experiences. How we view ourselves influences our
perceptions of others and, by extension, our
interactions with them.
The content in this category covers the notions of selfconcept and identity, along with the role of selfesteem, self-efficacy, and locus of control in the
development of self-concept. Identity formation,
© 2020 Association of American Medical Colleges

92

Self-Concept, Self-Identity, and Social Identity (PSY,
SOC)
▪ The role of self-esteem, self-efficacy, and locus of
control in self-concept and self-identity (PSY)
▪ Different types of identities (e.g., race/ethnicity,
gender, age, sexual orientation, class)
Formation of Identity (PSY, SOC)
▪ Theories of identity development (e.g., gender,
moral, psychosexual, social)
▪ Influence of social factors on identity formation
o Influence of individuals (e.g., imitation, lookingglass self, role-taking)
o Influence of groups (e.g., reference group)
▪ Influence of culture and socialization on identity
formation

including developmental stages and the social factors
that affect identity formation, is also covered here.
Theories are included to provide historical context for
the field of identity formation.

8B: Social thinking

Attributing Behavior to Persons or Situations (PSY)

Social thinking refers to the ways we view others and
our environment, as well as how we interpret others’
behaviors. A variety of factors ― personality,
environment, and culture ― factor into the beliefs and
attitudes we develop.

▪ Attributional processes (e.g., fundamental
attribution error, role of culture in attributions)
▪ How self-perceptions shape our perceptions of
others
▪ How perceptions of the environment shape our
perceptions of others

Our beliefs and attitudes about others and the
environment also shape the way we interact with each
other. To interact with others, we need to interpret
different aspects of a situation, including our
perception of ourselves, the behavior of others, and
the environment.
The content in this category covers our attitudes
about others and how those attitudes develop,
including how perceptions of culture and environment
affect attributions of behavior. It also covers how our
attitudes about different groups ― prejudice,
stereotypes, stigma, and ethnocentrism ― may
influence our interactions with group members.

Prejudice and Bias (PSY, SOC)
▪ Processes that contribute to prejudice
o Power, prestige, and class (SOC)
o The role of emotion in prejudice (PSY)
o The role of cognition in prejudice (PSY)
▪ Stereotypes
▪ Stigma (SOC)
▪ Ethnocentrism (SOC)
o Ethnocentrism vs. cultural relativism
Processes Related to Stereotypes (PSY)
▪ Self-fulfilling prophecy
▪ Stereotype threat

8C: Social interactions

Elements of Social Interaction (PSY, SOC)

Humans are social beings by nature. Though the
sentiment is simple, the actions and processes
underlying and shaping our social interactions are not.

▪ Status (SOC)
o Types of status (e.g., achieved, ascribed)
▪ Role
o Role conflict and role strain (SOC)
o Role exit (SOC)
▪ Groups
o Primary and secondary groups (SOC)
o In-group vs. out-group

The changing nature of social interaction is important
for understanding the mechanisms and processes
through which people interact with each other, both
individually and within groups. A variety of factors ―

© 2020 Association of American Medical Colleges

93

environment, culture, and biology ― affect how we
present ourselves to others and how we treat others.
For example, perceptions of prejudice and stereotypes
can lead to acts of discrimination, whereas positive
attitudes about others can lead to the provision of
help and social support.
The content in this category covers the mechanisms of
self-presentation and social interaction including
expressing and detecting emotion, impression
management, communication, the biological
underpinning of social behavior, and discrimination.

o Group size (e.g., dyads, triads) (SOC)
▪ Networks (SOC)
▪ Organizations (SOC)
o Formal organization
o Bureaucracy
▪ Characteristics of an ideal bureaucracy
▪ Perspectives on bureaucracy (e.g., iron law of
oligarchy, McDonaldization)
Self-Presentation and Interacting With Others (PSY,
SOC)
▪ Expressing and detecting emotion
o The role of gender in the expression and
detection of emotion
o The role of culture in the expression and
detection of emotion
▪ Presentation of self
o Impression management
o Front-stage vs. back-stage self (dramaturgical
approach) (SOC)
▪ Verbal and nonverbal communication
▪ Animal signals and communication (PSY, BIO)
Social Behavior (PSY)
▪ Attraction
▪ Aggression
▪ Attachment
▪ Altruism
▪ Social support (PSY, SOC)
▪ Biological explanations of social behavior in animals
(PSY, BIO)
o Foraging behavior (BIO)
o Mating behavior and mate choice
o Applying game theory (BIO)
o Altruism
o Inclusive fitness (BIO)

© 2020 Association of American Medical Colleges

94

Discrimination (PSY, SOC)
▪ Individual vs. institutional discrimination (SOC)
▪ The relationship between prejudice and
discrimination
▪ How power, prestige, and class facilitate
discrimination (SOC)

© 2020 Association of American Medical Colleges

95

Psychological, Social, and Biological Foundations of Behavior
Foundational Concept 9
Cultural and social differences influence well-being.
Social structure and demographic factors influence people’s health and well-being. Knowledge about basic
sociological theories, social institutions, culture, and demographic characteristics of societies is important to
understand how these factors shape people’s lives and their daily interactions.
Foundational Concept 9 focuses on social variables and processes that influence our lives.
Content Categories
▪
▪

Category 9A focuses on the link between social structures and human interactions.
Category 9B focuses on the demographic characteristics and processes that define a society.

With these building blocks, medical students will be able to learn about the ways patients’ social and
demographic backgrounds influence their perception of health and disease, the health care team, and
therapeutic interventions.
9A: Understanding social structure

Theoretical Approaches (SOC)

Social structure organizes all human societies.
Elements of social structure include social institutions
and culture. These elements are linked in a variety of
ways and shape our experiences and interactions with
others ― a process that is reciprocal.

▪ Microsociology vs. macrosociology
▪ Functionalism
▪ Conflict theory
▪ Symbolic interactionism
▪ Social constructionism
▪ Exchange-rational choice
▪ Feminist theory

The content in this category provides a foundation for
understanding social structure and the various forms
of interactions within and among societies. It includes
theoretical approaches to studying society and social
groups, specific social institutions relevant to student
preparation for medical school, and the construct of
culture.

© 2020 Association of American Medical Colleges

96

Social Institutions (SOC)
▪ Education
o Hidden curriculum
o Teacher expectancy
o Educational segregation and stratification
▪ Family (PSY, SOC)
o Forms of kinship (SOC)
o Diversity in family forms
o Marriage and divorce
o Violence in the family (e.g., child abuse, elder
abuse, spousal abuse) (SOC)

▪ Religion
o Religiosity
o Types of religious organizations (e.g., churches,
sects, cults)
o Religion and social change (e.g., modernization,
secularization, fundamentalism)
▪ Government and economy
o Power and authority
o Comparative economic and political systems
o Division of labor
▪ Health and medicine
o Medicalization
o The sick role
o Delivery of health care
o Illness experience
o Social epidemiology
Culture (PSY, SOC)
▪ Elements of culture (e.g., beliefs, language, rituals,
symbols, values)
▪ Material vs. symbolic culture (SOC)
▪ Culture lag (SOC)
▪ Culture shock (SOC)
▪ Assimilation (SOC)
▪ Multiculturalism (SOC)
▪ Subcultures and countercultures (SOC)
▪ Mass media and popular culture (SOC)
▪ Evolution and human culture (PSY, BIO)
▪ Transmission and diffusion (SOC)
9B: Demographic characteristics and processes

Demographic Structure of Society (PSY, SOC)

To understand the structure of a society, it is
important to understand the demographic
characteristics and processes that define it.
Knowledge of the demographic structure of societies
and an understanding of how societies change help us
comprehend the distinct processes and mechanisms
through which social interaction occurs.

▪ Age
o Aging and the life course
o Age cohorts (SOC)
o Social significance of aging
▪ Gender
o Sex vs. gender
o The social construction of gender (SOC)
o Gender segregation (SOC)

© 2020 Association of American Medical Colleges

97

The content in this category covers the important
demographic variables at the core of understanding
societies and includes concepts related to
demographic shifts and social change.

▪ Race and ethnicity (SOC)
o The social construction of race
o Racialization
o Racial formation
▪ Immigration status (SOC)
o Patterns of immigration
o Intersections with race and ethnicity
▪ Sexual orientation
Demographic Shifts and Social Change (SOC)
▪ Theories of demographic change (e.g., Malthusian
theory and demographic transition)
▪ Population growth and decline (e.g., population
projections, population pyramids)
▪ Fertility, migration, and mortality
o Fertility and mortality rates (e.g., total, crude,
age-specific)
o Patterns in fertility and mortality
o Push and pull factors in migration
▪ Social movements
o Relative deprivation
o Organization of social movements
o Movement strategies and tactics
▪ Globalization
o Factors contributing to globalization (e.g.,
communication technology, economic
interdependence)
o Perspectives on globalization
o Social changes in globalization (e.g., civil unrest,
terrorism)
▪ Urbanization
o Industrialization and urban growth
o Suburbanization and urban decline
o Gentrification and urban renewal

© 2020 Association of American Medical Colleges

98

Psychological, Social, and Biological Foundations of Behavior
Foundational Concept 10
Social stratification and access to resources influence well-being.
Social stratification and inequality affect all human societies and shape the lives of all individuals by affording
privileges to some and positioning others at a disadvantage.
Foundational Concept 10 focuses on the aspects of social inequality that influence how we interact with one
another, as well as how we approach our health and the health care system.
Content Category
▪

Category 10A focuses on a broad understanding of social class, including theories of stratification,
social mobility, and poverty.

With these building blocks, medical students will be able to learn about the ways social and economic factors
can affect access to care and the probability of maintaining health and recovering from disease.
10A: Social inequality

Spatial Inequality (SOC)

Barriers to access to institutional resources exist for
the segment of the population that is disenfranchised
or lacks power within a given society. Barriers to
access might include language, geographic location,
socioeconomic status, immigration status, and
racial/ethnic identity. Institutionalized racism and
discrimination are also factors that prevent some
groups from obtaining equal access to resources. An
understanding of the barriers to access to institutional
resources, informed by perspectives such as social
justice, is essential to address health and health care
disparities.

▪ Residential segregation
▪ Neighborhood safety and violence
▪ Environmental justice (location and exposure to
health risks)

The content in this category covers spatial inequality,
the structure and patterns of social class, and health
disparities in relation to class, race/ethnicity, and
gender.

© 2020 Association of American Medical Colleges

99

Social Class (SOC)
▪ Aspects of social stratification
o Social class and socioeconomic status
o Class consciousness and false consciousness
o Cultural capital and social capital
o Social reproduction
o Power, privilege, and prestige
o Intersectionality (e.g., race, gender, age)
o Socioeconomic gradient in health
o Global inequalities
▪ Patterns of social mobility
o Intergenerational and intragenerational mobility
o Vertical and horizontal mobility
o Meritocracy

▪ Poverty
o Relative and absolute poverty
o Social exclusion (segregation and isolation)
Health Disparities (SOC) (e.g., class, gender, and race
inequalities in health)
Health Care Disparities (SOC) (e.g., class, gender, and
race inequalities in health care)

© 2020 Association of American Medical Colleges

100

Critical Analysis and Reasoning Skills
What Will the Critical Analysis and Reasoning Skills Section Test?
The Critical Analysis and Reasoning Skills section of the MCAT exam will be similar to many of the verbal
reasoning tests you have taken in your academic career. It includes passages and questions that test
your ability to understand what you read. You may find this section unique in several ways, though,
because it has been developed specifically to measure the analysis and reasoning skills you will need to
be successful in medical school. The Critical Analysis and Reasoning Skills section achieves this goal by
asking you to read and think about passages from a wide range of disciplines in the social sciences and
humanities, followed by a series of questions that lead you through the process of comprehending,
analyzing, and reasoning about the material you have read.
Critical Analysis and Reasoning Skills passages are relatively short, typically between 500 and 600 words,
but they are complex, often thought-provoking pieces of writing with sophisticated vocabulary and, at
times, intricate writing styles. Everything you need to know to answer the test questions is in the
passages and the questions themselves. No additional coursework or specific knowledge is required to
do well on the Critical Analysis and Reasoning Skills section, but you, as the test taker, may find yourself
needing to read the passages and questions in ways that are different from the reading required in the
textbooks you used in most prehealth courses or on tests like the SAT Critical Reading exam. Passages
for the Critical Analysis and Reasoning Skills section — even those written in a conversational or
opinionated style — are often multifaceted and focus on the relationships between ideas or theories.
The questions associated with the passages will require you to assess the content, but you will also need
to consider the authors’ intentions and tones and the words they used to express their points of view.
This section is designed to:
▪
▪
▪

Test your comprehension, analysis, and reasoning skills by asking you to critically analyze
information provided in passages.
Include content from ethics, philosophy, studies of diverse cultures, population health, and a
wide range of social sciences and humanities disciplines.
Provide all the information you need to answer questions in the passages and questions
themselves.

Test Section

Number of Questions

Time

Critical Analysis and Reasoning
Skills

53

90 minutes

(note that questions are all
passage-based)

© 2020 Association of American Medical Colleges

101

Critical Analysis and Reasoning Skills Distribution of Questions by Critical Analysis and
Reasoning Skill and Passage Content in the Humanities and Social Sciences
You may wonder how many questions you’ll get testing a particular critical analysis and reasoning skill or
how many humanities or social science passages you’ll see on the test. The questions you see are likely
to be distributed in the ways described below. (These percentages have been approximated to the
nearest 5% and will vary from one test to another for a variety of reasons, including, but are not limited
to, controlling for question difficulty, using groups of questions that depend on a single passage, and
using unscored field-test questions on each test form.)
Critical Analysis and Reasoning Skill:
▪
▪
▪

Foundations of Comprehension, 30%
Reasoning Within the Text, 30%
Reasoning Beyond the Text, 40%

Passage Content:
▪
▪

Humanities, 50%
Social Sciences, 50%

What Is the Content of the Passages in the Critical Analysis and Reasoning Skills Section?
Passages in the Critical Analysis and Reasoning Skills section are excerpted from the kinds of books,
journals, and magazines that college students are likely to read. Passages from the social sciences and
humanities disciplines might present interpretations, implications, or applications of historical accounts,
theories, observations, or trends of human society as a whole, specific population groups, or specific
countries.
Of these two types of passages (social sciences and humanities), social sciences passages tend to be
more factual and scientific in tone. For example, a social sciences passage might discuss how basic
psychological and sociological assumptions help scholars reconstruct patterns of prehistoric civilizations
from ancient artifacts. Humanities passages often focus on the relationships between ideas and are
more likely to be written in a conversational or opinionated style. Therefore, you should keep in mind
the tone and word choice of the author in addition to the passage assertions themselves. Humanities
passages might describe the ways art reflects historical or social change or how the philosophy of ethics
has adapted to prevailing technological changes.
Critical Analysis and Reasoning Skills passages come from a variety of humanities and social sciences
disciplines.

© 2020 Association of American Medical Colleges

102

Humanities
Passages in the humanities are drawn from a variety of disciplines, including (but not limited to):
▪
▪
▪
▪
▪
▪
▪
▪
▪
▪
▪

Architecture
Art
Dance
Ethics
Literature
Music
Philosophy
Popular Culture
Religion
Theater
Studies of Diverse Cultures†

Social Sciences
Social sciences passages are also drawn from a variety of disciplines, including (but not limited to):
▪
▪
▪
▪
▪
▪
▪
▪
▪
▪
▪
▪

Anthropology
Archaeology
Economics
Education
Geography
History
Linguistics
Political Science
Population Health
Psychology
Sociology
Studies of Diverse Cultures

†

Depending on the focus of the text, a Studies of Diverse Cultures passage could be classified as
belonging to either the Humanities or Social Sciences.
© 2020 Association of American Medical Colleges

103

What Kinds of Analysis Skills Does the Critical Analysis and Reasoning Skills Section Require?
The Critical Analysis and Reasoning Skills section assesses three broad critical analysis and reasoning
skills. Questions in this section will ask you to determine the overall meaning of the text, to summarize,
evaluate, and critique the “big picture,” and to synthesize, adapt, and reinterpret concepts you
processed and analyzed. The questions that follow Critical Analysis and Reasoning Skills passages lead
you through this complex mental exercise of finding meaning within each text and then reasoning
beyond the text to expand the initial meaning. The analysis and reasoning skills you will be tested on
mirror those that mature readers use to make sense of complex materials. The skills assessed in the
Critical Analysis and Reasoning Skills section are listed below, and each skill is explained in the following
sections.

Critical Analysis and Reasoning Skills
Foundations of Comprehension
▪ Understanding the basic components of the text
▪ Inferring meaning or intent from immediate sentence context
Reasoning Within the Text
▪ Integrating distant components of the text to infer an author's message, intent, purpose, belief,
position, bias, assumptions
▪ Recognizing and evaluating arguments and their structural elements (claims, evidence, support,
relations)
Reasoning Beyond the Text
▪ Applying or extrapolating ideas from the passage to new contexts
▪ Assessing the impact of incorporating new factors, information, or conditions on ideas from the
passage

Foundations of Comprehension
The topics of some passages in the Critical Analysis and Reasoning Skills section will be familiar; some
will not. Explanations, illustrative examples, and definitions of significant specialized terms in these
passages will help you develop the strong basic foundation needed for answering all the questions you
encounter in this section of the MCAT exam. Questions that test Foundations of Comprehension rely on
many of the same activities required for Reading Within the Text questions. One key difference is in the
scope of the information needed to answer the question. The Foundations of Comprehension questions
mainly focus on inferring meaning or intent from an immediate sentence context.
Additionally, some questions may ask you about the overall meaning of information in the passages or
the author’s central themes or ideas; others may ask you to select the definitions of specific words or
phrases as they are used in context. These kinds of questions help you build the foundation that will
allow you to think in new ways about concepts or facts presented in the passages. Paragraph numbers
may be included in questions to help you locate relevant portions of the text.
Two sets of skills are the basis of the Foundations of Comprehension questions on the Critical Analysis
and Reasoning Skills section.
© 2020 Association of American Medical Colleges

104

Understanding the Basic Components of the Text
The most fundamental questions on the Critical Analysis and Reasoning Skills section ask about the basic
components of the passages. Comprehension questions at this level may ask you to provide a general
overview of the passage or to focus on specific portions of the text. You may be asked to recognize the
literal meaning of a particular word or phrase. You may be asked to identify the author’s thesis, the main
point or theme of the passage, or specific examples. In responding to these questions, you need to be
able to recognize the purpose of particular sentences and rhetorical labels such as “for example,”
“therefore,” or “consequently.”
Inferring Meaning or Intent From Immediate Sentence Context
Questions may also require you to infer meanings that can’t be determined from a literal reading of the
text, such as meanings the author has implied but did not state directly. Comprehension questions at
this level may ask you to interpret the meaning of words or expressions, or the author's intent, using the
immediate sentence context. These questions may ask you to interpret rhetorical devices or word
choice. Or, you may have to consider how the author has structured the text — for example, through
cause-and-effect relationships for discussions in the behavioral sciences, chronologically for historical
discussions, or point-and-counterpoint for political science pieces. Identifying the basic structure should
help you understand the passage and determine its general purpose.
You may also need to attend to specific subtle and nuanced rhetorical decisions an author has made to
shape his or her ideas, arguments, or discussions and perhaps to complicate a passage’s meaning. For
example, questions may ask you to explain a highlighted word or phrase or an unexpected transition in
ideas. To answer these questions, look for clues in the context around the specific sections of the
passage. An author’s choice about tone (e.g., humorous, authoritative, satirical) also contributes to — or
obscures — meaning, and tone can often communicate the purpose for which a passage is written (e.g.,
to persuade, instruct, inform, entertain). For example, a satirical piece may at first seem merely
entertaining, but a closer examination often reveals that its purpose is actually to persuade.
Some questions at this level may ask about information not specifically stated in the passage, and you
must make assumptions based on what the author merely hints at through his or her use of connotative
language or figures of speech.
The beginning and ending of passages are two specific sections where the author often provides
important information about the general theme, message, or purpose for the work. Does the author
state their main point in an introductory or closing sentence? Does the passage end with a definitive
solution, a partial resolution, or a call for additional research? Does it end with a dramatic rhetorical
statement or a joke that leaves unanswered questions? Again, considering these specific sections can
help inform your basic understanding of the passage.

Reasoning Within the Text
Questions that test Reasoning Within the Text rely on many of the same activities required for
Foundations of Comprehension questions. One key difference is in the scope of the information needed
to answer the question. The Foundations of Comprehension questions mainly focus on inferring
© 2020 Association of American Medical Colleges

105

meaning or intent from an immediate sentence context. Questions that test Reasoning Within the Text
differ from those assessing Foundations of Comprehension in that they ask you to integrate distant
passage components into a more generalized and complex interpretation of passage meaning.
It’s important to remember that Reasoning Within the Text questions do not ask you to provide your
own personal opinion. You may, in fact, disagree with the author’s overall conclusion yet find that the
conclusion is a reasonable inference from the limited information provided in the passage. If you
happen to know some obscure fact or anecdote outside the scope of the passage that could invalidate
the author’s conclusion, ignore it. The content of the passage or new information introduced by the
questions should be the only sources you base your responses on.
Two sets of skills are the basis of the Reasoning Within the Text questions on the Critical Analysis and
Reasoning Skills section.
Integrating Distant Components of the Text
Many questions that test Reasoning Within the Text skills require you to integrate distant components
of the text to infer meaning or intent. You may be asked to determine an author's message, purpose,
position, or point of view. This may also extend to inferring their beliefs, noticing their assumptions, and
detecting bias. When it is not directly stated in any single sentence, you may be asked to infer what the
author’s main thesis might be. You may be asked to consider whether each section of text contributes to
a sustained train of thought, as opposed to presenting an isolated detail or digressing from the central
theme. You may be asked about paradoxes, contradictions, or inconsistencies that can be detected
across different parts of the passage. You will also need to be able to recognize when an author presents
different points of view within the passage.
To infer the author’s beliefs, attitudes, or bias, look for clues in the tone of the passage, in the author’s
use of language or imagery, and in the author’s choice of sources. To determine the author’s position,
look for their expressed point of view. Carefully consider the extent to which the author uses summaries
or paraphrases to introduce others’ points of view. It’s very important to attend to perspective: Does
the author present their own perspective, or do they use verbatim quotations or restatements from the
perspective of other sources? You may be asked to identify points of view, other than the author’s,
presented indirectly through the author’s summaries or paraphrases.
Recognizing and Evaluating Arguments
Questions assessing Reasoning Within the Text will also require you to understand how the different
parts of the passage fit together to support the author’s central thesis. Some questions will direct your
attention to an argument, claim, or evidence presented in the passage and then ask you to evaluate it
according to specific criteria. The criteria could be the logic and plausibility of the passage text, the
soundness of its arguments, the reasonableness of its conclusions, the appropriateness of its
generalizations, or the credibility of the sources the author cites. The questions require you to dig
beneath the passage’s surface as you examine the presence or absence of evidence, the relevance of
information, and faulty notions of causality and to determine the significance of and relationships
among different parts of a passage. Some questions may require that you analyze the author’s language,
© 2020 Association of American Medical Colleges

106

stance, and purpose. For example, plausible-sounding transitional phrases may in fact be tricky. If read
quickly, the words appear to make a legitimate connection between parts of a passage; however, when
subjected to scrutiny, the links they appear to have established may fall apart.
The skills required to answer both types of Reasoning Within the Text questions may sound like a long
list of possible critical and analysis skills to have mastered, but they are skills you probably already have
and use every day. Similar to your reactions when you hear someone trying to convince you about
something, persuade you to think a particular way, or sell you something, these questions often invite
you to doubt and then judge the author’s intentions and credibility. Questioning an author is a
legitimate and often necessary analysis strategy that can serve test takers well when making sense of
complex text. Answering these questions requires looking beyond contradictions or omission of facts or
details to find clues such as vague or evasive terms or language that sounds self-aggrandizing,
overblown, or otherwise suspect within the context of the passage. Credible sources — essayists,
scientists, lecturers, even pundits — should be both authoritative and objective and should clearly
demonstrate expertise. Blatant, one-sided arguments and rigid points of view are easy to identify, but
some authors are more nuanced in presenting biased ideas in the guise of objectivity. The key to
identifying bias lies in identifying the author’s treatment of ideas, which you achieve by analyzing and
evaluating different aspects of the passage. For example, an author who uses demeaning stereotypes or
derogatory labels is not likely to be a source of objective, judicious analysis.

Reasoning Beyond the Text
The final category, Reasoning Beyond the Text, requires you to use one of two analysis or reasoning
skills, which in a way can be thought of as two sides of a single coin. Questions assessing the first set of
skills ask you to apply or extrapolate information or ideas presented in the passage to a new or novel
situation — for example, extending information the author presents beyond the actual context of the
passage.
The second set of skills involves considering new information presented in a test question, mentally
integrating this new information into the passage content, and then assessing the potential impact of
introducing the new elements into the actual passage. Reasoning about new, hypothetical elements
should cause you to synthesize passage content anew and alter your interpretation of the passage in
some plausible way.
Application and integration questions elicit some of the same kinds of thinking. Both types deal with
changes caused by combinations or comparisons, and both test your mental flexibility. They do differ,
however, and their distinct requirements are explained in more detail below. Remember, though, that
as with questions assessing different levels of analysis and reasoning, you must still use only the content
of the passages and the new information in the questions to determine your answers. Keep avoiding the
temptation to bring your existing knowledge to bear in answering these questions.

© 2020 Association of American Medical Colleges

107

Applying or Extrapolating Ideas From the Passage to New Contexts
Virtually all questions assessing application or extrapolation skills ask you how the information or ideas
presented in the passage could be extended to other areas or fields. This is the kind of high-level
analysis and reasoning skill scientists or theoreticians use when they consider a set of facts or beliefs
and create new knowledge by combining the “givens” in new ways. Of course, these combinations may
or may not result in a successful combination or outcome.
For each application question, the passage material is the “given,” and the test question provides
specific directions about how the passage information might be applied to a new situation or how it
might be used to solve a problem outside the specific context of the passage. As the test taker, your first
task is to analyze the choices offered in the four response options so that you can gauge the likely
outcome of applying the existing passage content to the specified new context. Each response option
will yield a different result, but each test question has only one defensible and demonstrably correct
response option.
The correct answer is the one option that presents the most likely and most reasonable outcome, based
only on the information provided in the passage and the question. The questions do not assess your
personal ability to apply information or solve problems, only your ability to apply information from the
question to the passage you have read. For example, if a question asks you to determine the author’s
likely response to four hypothetical situations, you would choose the response most consistent with
what the author has already said or done according to the text of the passage. In determining the
correct response, rule out the options that do not fit or are incongruent with the context (e.g.,
framework, perspective, scenario) created by the passage material.
Application questions sometimes require selecting a response option that is most analogous to some
relationship in the passage. Here the parameters are broad. Likeness is measured not by inherent
similarity but by analogy. Questions dealing with analogies test the ability to identify a fundamental
common feature that seemingly different things or processes share. This may sometimes require
translating a figurative comparison into equivalent sets of literal terms. However, the task always
requires looking beneath surface imagery to discern underlying relationships or paradigms.
Assessing the Impact of Incorporating New Factors, Information, or Conditions on
Ideas From the Passage
The essential difference between application and incorporation skills is that the two-part purpose of
incorporation questions is to introduce a specific piece of information for you to consider and ask you to
assess how ideas in the passage might be affected by its introduction. The premise of these questions is
that ideas and information in the passages are potentially malleable, not a fixed framework, as in
application questions.
In some incorporation questions, you must find the best answer to a “what if” question by
reinterpreting and reassessing passage content with the additional fact or idea introduced by the
question. Does the new information support or contradict the inherent logic of the passage? Could the
new information coexist with what is already in the passage, or would it negate an aspect of the
© 2020 Association of American Medical Colleges

108

author’s argument? If the latter is the case, the question could ask what modifications or alterations
might need to be made to the passage content to accommodate the new element introduced by the
question. Remember, the passage should be considered malleable.
Other forms of incorporation questions may ask you to think about a possible logical relationship that
might exist between the passage content and the facts or assertions included in the answer options. The
task is to select the one option that, if added to the passage content, would result in the least amount of
change. The correct response option will present the situation or argument that is most similar to what
is outlined in the passage. In other words, you must determine which new fact or assertion would least
alter the central thesis the passage has developed.

© 2020 Association of American Medical Colleges

109

20-002D (10/20)

`

  // Load the pre-set text file when the API key is set
  useEffect(() => {
    if (!isApiKeySet) return;

    const loadDocument = async () => {
      setIsLoading(true);
      addMessage("Loading Index", 'bot');
      try {
        const response = await fetch('/Users/jvnk/Desktop/mymcat/chatbot-app/src/documents/3-2020_whats_on_the_mcat_content_outline.txt');
        if (!response.ok) {
          throw new Error('Failed to fetch the document');
        }
        var text = await response.text();
        text = textfi;
        console.log('Fetched text (first 100 chars):', text.slice(0, 100));
        // const text = await response.text();

        const splitter = new RecursiveCharacterTextSplitter({
          chunkSize: 10250,
          chunkOverlap: 1000
        });
        const chunks = await splitter.splitText(text);
        const docs = chunks.map(chunk => new Document({ pageContent: chunk }));

        const embeddings = new OpenAIEmbeddings({ openAIApiKey: apiKey });
        const store = await MemoryVectorStore.fromDocuments(docs, embeddings);
        setVectorStore(store);

        addMessage("Document processed successfully! You can now ask questions about the MCAT content outline.", 'bot');
      } catch (error) {
        console.error("Error loading document:", error);
        addMessage("There was an error loading the document. Please try again later.", 'bot');
      } finally {
        setIsLoading(false);
      }
    };
    loadDocument();
  }, [isApiKeySet, apiKey]);

  const addMessage = (text: string, sender: 'user' | 'bot') => {
    const newMessage: Message = {
      id: Date.now() + Math.random(),
      text,
      sender
    };
    setMessages(prevMessages => [...prevMessages, newMessage]);
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (input.trim() === '' || !isApiKeySet) return;
    
    const userMessage = input.trim();
    addMessage(userMessage, 'user');
    setInput('');
    
    setIsLoading(true);
    
    try {
      if (vectorStore) {
        const model = new ChatOpenAI({ 
          openAIApiKey: apiKey, 
          temperature: 0.7,
          modelName: "gpt-4o-mini-2024-07-18"
        });
        
        const retriever = vectorStore.asRetriever({k:7});
        
        const prompt = ChatPromptTemplate.fromTemplate(`
          You are a helpful assistant that answers questions about academic documents.
          
          Use the following context to answer the question. If the answer is not in the context, 
          say "I don't have enough information about that in the document."
          
          Context:
          {context}
          
          Question: {input}
          
          Answer:
        `);
        
        const documentChain = await createStuffDocumentsChain({
          llm: model,
          prompt,
        });
        
        const retrievalChain = await createRetrievalChain({
          combineDocsChain: documentChain,
          retriever,
        });
        
        const response = await retrievalChain.invoke({
          input: userMessage,
        });
        
        let botResponse = "I couldn't find a relevant answer in the document.";
        if ('answer' in response && typeof response.answer === 'string') {
          botResponse = response.answer;
        } else if ('output' in response && typeof response.output === 'string') {
          botResponse = response.output;
        } else if ('result' in response && typeof response.result === 'string') {
          botResponse = response.result;
        } else if (typeof response === 'string') {
          botResponse = response;
        } else if (response && typeof response === 'object') {
          const responseObj = response as Record<string, unknown>;
          const possibleAnswerKeys = Object.keys(responseObj).filter(key => 
            typeof responseObj[key] === 'string' && (responseObj[key] as string).length > 0
          );
          if (possibleAnswerKeys.length > 0) {
            botResponse = responseObj[possibleAnswerKeys[0]] as string;
          }
        }
        
        addMessage(botResponse, 'bot');
      } else {
        addMessage("Please wait while the document is being loaded.", 'bot');
      }
    } catch (error) {
      console.error("Error getting response:", error);
      addMessage("Sorry, I encountered an error while processing your question. Please try again.", 'bot');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isApiKeySet) {
    return (
      <div className="chat-container">
        <div className="chat-header">
          <h1>JV AND DH2 RAG Chatbot</h1>
        </div>
        <div className="api-key-form">
          <p>Please enter your OpenAI API key to continue:</p>
          <form onSubmit={handleApiKeySubmit}>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your OpenAI API key"
              className="api-key-input"
            />
            <button type="submit">Submit</button>
          </form>
          <p className="api-key-note">Note: Your API key is stored locally in your browser and never sent to our servers.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>RAG Chatbot</h1>
      </div>
      
      <div className="messages-container">
        {messages.map(message => (
          <div 
            key={message.id} 
            className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
          >
            {message.text}
          </div>
        ))}
        {isLoading && (
          <div className="message bot-message loading">
            <div className="loading-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form className="input-form" onSubmit={handleSendMessage}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask something about the document..."
          disabled={isLoading || !vectorStore}
        />
        <button type="submit" disabled={isLoading || !vectorStore}>
          Send
        </button>
      </form>
    </div>
  );
};

export default App;