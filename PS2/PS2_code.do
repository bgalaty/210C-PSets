/**************************************************************
ECON 210C: Homework 2
**************************************************************/


/**************************************************************
Preamble
**************************************************************/{
clear
// cd "/Users/bridgetgalaty/Documents/*School/PhD/FirstYear/210C/"
local datain "/Users/bridgetgalaty/Documents/*School/PhD/FirstYear/210C/PS2-Johannes/data"
local results "/Users/bridgetgalaty/Documents/*School/PhD/FirstYear/210C/GitFolder/Homework/PS2"

// ssc install blindschemes, replace
// ssc install GRSTYLE, replace
// ssc install freduse, replace
// ssc install palettes, replace
// ssc install colrspace

* Import data method
global withfreduse = 0
	
*Establish color scheme
global bgcolor "211 216 221"
global fgcolor "0 63 125"
	/* Decomp Colors */
global color1 "157 98 209"
global color2 "168 3 42"
global color3 "102 38 93"
		
* Establish some graphing setting
graph drop _all
set scheme plotplainblind // Biscoff
grstyle init
// Legend settings
grstyle set legend 6, nobox
// General plot 
grstyle color background "${bgcolor}"
grstyle color major_grid "${fgcolor}"
grstyle set color "${fgcolor}": axisline major_grid // set axis and grid line color
grstyle linewidth major_grid thin
grstyle yesno draw_major_hgrid yes
grstyle yesno grid_draw_min yes
grstyle yesno grid_draw_max yes
grstyle anglestyle vertical_tick horizontal
}

/**************************************************************
Data Setup
**************************************************************/{
clear
/* Import Romer Data */ {
	use `datain'/Monetary_shocks/RR_monetary_shock_quarterly.dta
	rename date dateq
	format dateq %tq 

	tempfile romer
	save `romer', replace 
	
}

/* Get data from FRED  */ {
local tsvar "FEDFUNDS UNRATE GDPDEF USRECM"
foreach v of local tsvar {
	import delimited using `datain'/`v'.csv, clear case(preserve)
	rename DATE date
	tempfile `v'_dta
	save ``v'_dta', replace
		}
use `FEDFUNDS_dta', clear
	keep date
foreach v of local tsvar {
	joinby date using ``v'_dta', unm(b)
	drop _merge
		}
	}

/* Clean data */{
gen daten = date(date, "YMD")
format daten %td
drop if yofd(daten) < 1960  | yofd(daten) > 2023 // data is per quarter in 1947 but per month after 
gen INFL = 100*(GDPDEF - GDPDEF[_n-12])/GDPDEF[_n-12] //year to year inflation
	la var INFL "Inflation Rate"
	la var FEDFUNDS "Federal Funds Rate"
	la var UNRATE "Unemployment Rate"
	la var daten Date // re=label date 
local tsvar "FEDFUNDS UNRATE INFL USRECM" // Reset local varlist to include created inflation var

gen mdate = mofd(daten)
format mdate %tm 
	
gen qdate = qofd(daten)
format qdate %tq
	}
	
/* Format recession bars */{
egen temp1 = rowmax(`tsvar')
sum temp1
local max = ceil(r(max)/5)*5
generate recession = `max' if USREC == 1
drop temp1
egen temp1 = rowmin(`tsvar')
sum temp1
if r(min) < 0 {
	local min = ceil(abs(r(min))/5)*-5
}
if r(min) >= 0 {
	local min = floor(abs(r(min))/5)*5
}
	replace  recession = `min' if USREC == 0 //
drop temp1
la var recession "NBER Recessions"

}
}





/**************************************************************
Question 1.a
**************************************************************/{

/* Graph the data */ {
tsset daten
capture twoway (area recession daten, color("${fgcolor}") base(`min')) ///
	(tsline FEDFUNDS, lc("${color1}") lp(solid) lw(medthick)) || ///
	(tsline UNRATE, lc("${color2}") lp(dash) lw(medthick)) || ///
	(tsline INFL, lc("${color3}") lp(dot) lw(medthick)) || ///
	, ///
	title("Monthly U.S. Macroeconomic Indicators, 1960-2023", c("${fgcolor}")) ///
	tlabel(, format(%dCY) labc("${fgcolor}")) ttitle("") ///
	yline(0, lstyle(foreground) lcolor("${fgcolor}") lp(dash)) ///
	caption("Source: FRED." "Note: Shaded regions denote recessions.", c("${fgcolor}")) ///
	ytitle("Percent", c("${fgcolor}")) ///
	name(raw_data) ///
	legend(on order(2 3 4) pos(6) bmargin(tiny) r(1))  //bplacement(ne) 
	graph export `results'/1_a.pdf, replace
	}
}
	
/**************************************************************
Question 1.b
**************************************************************/{
local ImpulseVars "FEDFUNDS UNRATE INFL"
gen dateq = qofd(daten)
collapse (mean) `tsvar' (max) recession (last) date daten, by(dateq)
tsset dateq, quarterly
keep if (yofd(daten) >= 1960) & (yofd(daten) <= 2007)
var `ImpulseVars', lags(1/4)
	irf set var_results, replace
	irf create var_result, step(20) set(var_results) replace
	capture irf graph irf, impulse(`ImpulseVars') response(`ImpulseVars') byopts(yrescale) ///
		yline(0, lstyle(foreground) lcolor("${fgcolor}") lp(dash)) ///
		name(var_results)
	graph export `results'/1_b.pdf, replace
	
/// I don't know how to get this to go in the right order... I've tried rearranging to no avail
}

/**************************************************************
Question 1.d
**************************************************************/{
// /* Manual Choleshy Decomp */
// matrix A = (1,0,0 \ .,1,0 \ .,.,1)
// matrix B = (.,0,0 \ 0,.,0 \ 0,0,.)
// svar `ImpulseVars', lags(1/4) aeq(A) beq(B)
// irf create mysirf, set(mysirfs) step(20) replace
// irf graph sirf, impulse(`ImpulseVars') response(`ImpulseVars') ///
// 		yline(0, lstyle(foreground) lcolor("${fgcolor}") lp(dash)) ///
// 		name(svar_results_manual)

var `ImpulseVars', lags(1/4)
irf create myirf, set(myirfs) step(20) replace
capture irf graph oirf, impulse(`ImpulseVars') response(`ImpulseVars') ///
		yline(0, lstyle(foreground) lcolor("${fgcolor}") lp(dash)) ///
		name(svar_results_oirf)
	graph export `results'/1_d.pdf, replace
}

/**************************************************************
Question 1.f
*************************************************************/{
capture predict resid_monetary, residuals equation(FEDFUNDS)
tsset daten

tsset daten
capture twoway /// (area recession daten, vertical color("${fgcolor}") base(`min'-5)) ///
	(tsline resid_monetary, lc("${color1}") lp(solid)) ///
	, ///
	title("Identified Monetary Shock", c("${fgcolor}")) ///
	tlabel(, format(%dCY) labc("${fgcolor}")) ttitle("") ///
	yline(0, lstyle(foreground) lcolor("${fgcolor}") lp(dash)) ///
	caption("Source: FRED." "Note: Shaded regions denote recessions.", c("${fgcolor}")) ///
	ytitle("Residuals", c("${fgcolor}")) ///
	name(shock_time_series, replace) ///
// 	legend(on order(1 "Estimated Monetary Shock") 
	graph export "/Users/bridgetgalaty/Documents/*School/PhD/FirstYear/210C/GitFolder/Homework/PS2/1_f.pdf", replace
// 	graph export `results'/1_f.pdf, replace
}





/**************************************************************
Question 2.a
*************************************************************/{
/* Combine with Romer Data */{
merge m:1 dateq using `romer' , nogen

foreach x in  resid resid_romer resid_full{
	replace `x' = 0 if dateq <= tq(1969q1)
}
}
}

/**************************************************************
Question 2.b
*************************************************************/{
tsset dateq

var INFL UNRATE FEDFUNDS, lags(1/8) exog(L(0/12).resid_full)
irf create myrirf, step(20) replace
irf graph dm, impulse(resid_full) ///
		yline(0, lstyle(foreground) lcolor("${fgcolor}") lp(dash)) ///
		name(rr_var, replace)
// graph export `results'/2_b.pdf, replace
graph export "/Users/bridgetgalaty/Documents/*School/PhD/FirstYear/210C/GitFolder/Homework/PS2/2_b.pdf", replace
}

// response(`ImpulseVars') 

var INFL UNRATE FEDFUNDS, lags(1/8) exog(L(0/12).resid_full)
irf create myrirf, step(20) replace
irf graph dm, impulse(resid_full) irf(myrirf) byopts(title(VAR with 8 Lags and RR Shocks) yrescale) /// INFL UNRATE 
yline(0,  lcolor(black) lp(dash) lw(*2)) legend(col(2) order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall))  ///
name(var_results, replace )





