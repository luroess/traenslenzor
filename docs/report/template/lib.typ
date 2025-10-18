// Copyright 2024 Felix Schladt https://github.com/FelixSchladt

#import "@preview/hydra:0.6.2": hydra
#import "colors.typ": *

// Import libs
#import "libs/codelistings.typ": *
#import "libs/notes.typ": *
#import "libs/tablestyle.typ": *
#import "libs/requirements.typ": *
#import "libs/utils.typ": *
#import "libs/risk.typ": *

#import "titlepage.typ": *
#import "@preview/glossarium:0.5.9": *

#let std-bibliography = bibliography

#let hm-template(
  title: none,
  subtitle: none,
  top-remark: none,
  show-table-of-contents: true,
  toc-depth: 2,
  appendix: none,
  glossary: none,
  bibliography: none,
  bib-style: "ieee",
  version: "0.1",
  authors: "",
  date: datetime.today(),
  project_logo: none,
  project_logo_dimensions: (auto, auto),
  titlepage_logo: none,
  titlepage_logo_dimensions: (auto, auto),
  lastpage: none,
  text-size: 12pt, //textsize for non header & footer text
  body,
) = {
  // Default to subtitle but enable manual setting
  if top-remark == none {
    top-remark = subtitle
  }

  // Setup glossary
  show: make-glossary
  register-glossary(glossary)


  // Design  configurations
  let accent_line = line(length: 100%, stroke: (paint: hm_black, thickness: 1pt));

  // Fonts
  let body-font = "Roboto"
  let heading-font = "Roboto"

  let text-size-template = 10pt
  set text(font: body-font, lang: "de", text-size-template) //template text size
  set par(justify: true)
  show heading: set text(weight: "semibold", font: heading-font, fill: hm_grey_dark)

  set page(
    margin: (
      top: 6em, 
      bottom: 6em, 
      rest: 6em // Side margins
      ))

  titlepage(
    title: title,
    subtitle: subtitle,
    authors: authors,
    logo: titlepage_logo,
    logo_dimensions: titlepage_logo_dimensions,
    toc-depth: toc-depth,
    text-size: text-size-template,
  )
  
  set page(
    header: context {
        set text(text-size-template)
        grid(
          columns: (40%, 20%, 40%),
          align(left)[
            Modularbeit
          ],
          align(center)[
            #set align(bottom)
            // #set text(
            //   fill: hm_grey_dark,
            //   weight: "semibold",
            // )
            // #let headings = query(heading.where(level: 1))
            // #if headings.len() > 0 and not headings.any(it => it.location().page() == here().page() - 1) and here().page() > 1{
            //     hydra(1, skip-starting: true)
            // }
            #set image(height: 25pt)
            #image("assets/HM_Logo_RGB.png")
          ],
          align(right)[
            #set par(justify: false)
            #top-remark
          ]
        )
      
      accent_line
    }, 
    footer: context{
      //accent_line
      
      set text(text-size-template)
      grid(
        columns: (1fr, 1fr),
        align(left)[
          #if version != none [
            #authors #date.display("[year]")\
            Version #version
          ] else [
            #authors
          ]
          
        ],
        // align(center)[
        //   #text(
        //     fill: hm_grey_dark,
        //     weight: "semibold",
        //     title
        //   )
        // ],
        align(right)[
          #numbering(
            "1 / 1",
            ..counter(page).get(),
            ..counter(page).at(<end>),
          )
        ]
      )
    }
  )
  
  set math.equation(numbering: "1.")
  
  // Heading settings
  show heading.where(level: 1): it => {
    pagebreak()
    text(size: 20pt, it)
    v(1.25em)
  }
  show heading.where(level: 2): it => v(1em) + it + v(1em)
  show heading.where(level: 3): it => v(1em) + it + v(0.75em)

  set text(text-size)

  // --------- Space for Glossary Abstract etc ----------

  
  // Display glossary.
  if glossary != none {
    [= Glossary]
    // This uses Glossarium to print the glossary
    // for configuration please refer to glossarium documentation
    print-glossary(glossary, disable-back-references: true)
  }

  // ---------- Setup Chapter Headings -------------------

  // Do numbered headings
  set heading(numbering: "1.")

  // ----------- Setup Completed - Content ---------------


  body

  // ----------- Other stuff - Bib gloss appendix etc ----

  
  // Non numbered headings
  set heading(numbering: none)


  // Display bibliography.
  if bibliography != none {
    [= References]
    set std-bibliography(
      title: none,
      style: bib-style
      )
    bibliography
  }

  // Display appendix.
  if appendix != none {
    //heading(level: 1, numbering: none)[Appendix]
    [= Appendix]
    include appendix
  }

  // Last Page, possible for reference, versioning & contact information
  if lastpage != none {
    lastpage
  }

  [#metadata(none)<end>]
}