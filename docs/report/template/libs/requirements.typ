// Copyright 2024 Felix Schladt https://github.com/FelixSchladt

#import "@preview/drafting:0.2.0": *
#import "@preview/glossarium:0.5.9": *
#import "/template/libs/utils.typ": *


// ----------------------------------------------------------------------------------
// Content -> String
// This is required in case acronyms are used in requirement titles. 
// As content is not usable for label creation and the acronym should inserted, 
// converting content into str is a good solution.
// Code from: https://sitandr.github.io/typst-examples-book/book/typstonomicon/extract_plain_text.html
// ----------------------------------------------------------------------------------

#let stringify-by-func(it) = {
  let func = it.func()
  return if func in (parbreak, pagebreak, linebreak) {
    "\n"
  } else if func == smartquote {
    if it.double { "\"" } else { "'" } // "
  } else if it.fields() == (:) {
    // a fieldless element is either specially represented (and caught earlier) or doesn't have text
    ""
  } else {
    panic("Not sure how to handle type `" + repr(func) + "`")
  }
}

#let plain-text(it) = {
  return if type(it) == str {
    it
  } else if it == [ ] {
    " "
  } else if it.has("children") {
    it.children.map(plain-text).join()
  } else if it.has("body") {
    plain-text(it.body)
  } else if it.has("text") {
    if type(it.text) == str {
      it.text
    } else {
      plain-text(it.text)
    }
  } else {
    // remove this to ignore all other non-text elements
    stringify-by-func(it)
  }
}


// ----------------------------------------------------------------------------------
// Label Registry
// ----------------------------------------------------------------------------------

#let label_registry = state("label_reg", ())

#let add_req_label(title) = {
  context {
    let name = "req_" + plain-text(title).replace(" ", "_")
  
    let dup = false
    
      let labels = label_registry.get()
      while name in labels {
        if dup {
          name = name.trim("_" + name.last(), at: end, repeat: false) + "_" + str(int(name.last()) + 1)
          continue
        }
        name = name + "_1"
        dup = true
    }
  
    label_registry.update(
      label_reg => {
        label_reg.push(name.replace(" ", "_"))
        label_reg
      } 
    )
  }
}

#let show_req_labels() = {
  context {
    let labels = label_registry.get()
    for label in labels {
      label + " "
    }
  }
}

#let get_latest_req_label() = {
  let labels = label_registry.get()
  str(labels.last())
}


// ----------------------------------------------------------------------------------
// Requirements
// ----------------------------------------------------------------------------------

#let requirements(
  functional_chapter_description: str,
  functional:    (
    (
      title: str, 
      description: str, 
      authors: (), 
      tracebility: str, 
      subrequirements: ()
    ),
  ), 
  non_functional_chapter_description: str,
  nonfunctional: (
    (
      title: str, 
      description: str, 
      authors: (), 
      tracebility: str, 
      subrequirements: ()
    ),
  )
) = {
  let requirement(
    title, 
    description, 
    functional, 
    tracebility: str, 
    authors: (), 
    subrequirements: ()
  ) = (
    title: title,
    authors: authors,
    description: description,
    tracebility: tracebility,
    functional: functional,
    subrequirements: subrequirements
  )

  let get_numbering(prenum, ctr, total_reqs_ctr) = {
    let ctr_dep = str(total_reqs_ctr).len()
    let ctr_str = str(ctr)
    while (ctr_str.len() != ctr_dep) {
      ctr_str = "0" + ctr_str
    }
    return prenum + ctr_str
  }

  let display_reqs(reqs, ctr, total_num, prenumbering) = {
    for req in  reqs{
      ctr += 1
      let numbering = get_numbering(prenumbering, ctr, total_num)

      // Assign label
      add_req_label(req.title)

      // Display requirement
      context [
        #show heading.where(level: 4): it => {
          block(it.body)
        }
        #set par(justify: false)
        #heading(level:4, supplement: none, "[" + numbering + "] " + req.title)
        #label(get_latest_req_label())
      ]
      
      // Show authors if provided
      if req.authors.len() > 0{
        authors_list(req.authors)
      }
      
      req.description

      linebreak()
      req.tracebility

      if req.subrequirements.len() > 0 {
        display_reqs(req.subrequirements, 0, req.subrequirements.len(), numbering + ".")
      }
    }
  }

  let convert_to_requirements(reqs) = {
    let out_reqs = ()
    for req in reqs {
      let subreq = ()
      if "subrequirements" in req {
        subreq = convert_to_requirements(req.at("subrequirements"))
      }

      let tracebility = ""
      if "tracebility" in req {
        tracebility = req.at("tracebility")
      }

      let authors = ()
      if "authors" in req {
        authors = req.at("authors")
      }
      
      let x = type(req)

      out_reqs.push(
        requirement(
          req.at("title"), 
          req.at("description"), 
          tracebility: tracebility, 
          false, 
          authors: authors, 
          subrequirements: subreq,
        )
      )
    }
    return out_reqs
  }

  let freqs = convert_to_requirements(functional)
  let nreqs = convert_to_requirements(nonfunctional)

  let ctr = 0
  for (title, lbl, description, requirements) in (
    (
      "Funktionale Anforderungen", 
      <req_funcional>, 
      functional_chapter_description, 
      freqs
    ), (
      "Nichtfunktionale Anforderungen", 
      <req_nonfunctional>, 
      non_functional_chapter_description, 
      nreqs
    )
  ){
    [#[== #title] #lbl]
    description

    display_reqs(requirements, ctr, nreqs.len() + freqs.len(), "R")
    
    ctr+= requirements.len()
  }
}
